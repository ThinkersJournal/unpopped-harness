// SPDX-License-Identifier: MIT OR Apache-2.0
//! Third target: a Slang candidate (`unpopped-slang`), compiled to SPIR-V by `slangc` and run
//! through `vulkane`, checked against the cpu-c reference over every f32.
//!
//! Same contract, [`Report`] and two-sentinel never-wrote test as the other targets: the loop is
//! `gpu::compare_with_runner`. Only the runner is Vulkan-specific.
//!
//! The emitted kernel has two storage buffers, `input0` at binding 0 and `output` at binding 1
//! (set 0, slangc's default assignment), `[numthreads(256, 1, 1)]`, and no bounds check, so every
//! dispatch covers a whole `GPU_CHUNK` and both buffers are exactly that long.

use std::path::{Path, PathBuf};
use std::process::Command;

use vulkane::safe::{
    AccessFlags, ApiVersion, Buffer, BufferCreateInfo, BufferUsage, CommandPool, ComputePipeline,
    DescriptorPool, DescriptorPoolSize, DescriptorSetLayout, DescriptorSetLayoutBinding,
    DescriptorType, DeviceCreateInfo, DeviceMemory, Fence, Instance, InstanceCreateInfo,
    MemoryPropertyFlags, PipelineLayout, PipelineStage, QueueCreateInfo, QueueFlags, ShaderModule,
    ShaderStageFlags,
};

use crate::gpu::{compare_with_runner, GPU_CHUNK};
use crate::{Kernel, Report, Tier};

const LOCAL_SIZE: u32 = 256;
const BYTES: u64 = (GPU_CHUNK * 4) as u64;

fn instance() -> Result<Instance, String> {
    Instance::new(InstanceCreateInfo {
        application_name: Some("unpopped-harness"),
        api_version: ApiVersion::V1_0,
        ..Default::default()
    })
    .map_err(|e| format!("vulkan instance: {e}"))
}

/// Names of the Vulkan devices with a compute queue, in the order `device_index` counts them.
pub fn vulkan_devices() -> Result<Vec<String>, String> {
    let inst = instance()?;
    let all = inst
        .enumerate_physical_devices()
        .map_err(|e| format!("enumerate: {e}"))?;
    Ok(all
        .iter()
        .filter(|pd| pd.find_queue_family(QueueFlags::COMPUTE).is_some())
        .map(|pd| pd.properties().device_name().to_string())
        .collect())
}

/// The stamp a record made on Vulkan device `device_index` (see [`vulkan_devices`]) carries: the
/// device, the API version, the driver version and vendor as the driver reports them, and the
/// `slangc` arguments the kernel was built with. A record is a measurement of a driver and a flag
/// set, not of a chip: a new driver or a different argument set is a different stamp, and Unpopped
/// then applies no record. `target` is the Unpopped target token the cell is keyed to.
pub fn device_stamp(
    device_index: usize,
    target: unpopped_vocab::TargetId,
    slangc_args: &[&str],
) -> Result<unpopped::exactness::DeviceStamp, String> {
    let inst = instance()?;
    let physical = inst
        .enumerate_physical_devices()
        .map_err(|e| format!("enumerate: {e}"))?
        .into_iter()
        .filter(|pd| pd.find_queue_family(QueueFlags::COMPUTE).is_some())
        .nth(device_index)
        .ok_or_else(|| format!("no compute-capable Vulkan device at index {device_index}"))?;
    let props = physical.properties();
    let captured_unix_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    Ok(unpopped::exactness::DeviceStamp::new(
        unpopped_vocab::HwStamp {
            target,
            device_name: props.device_name().to_string(),
            runtime_version: format!("{:?}", props.api_version()),
            captured_unix_s,
        },
        format!(
            "vendor={:#x} driver={:#x}",
            props.vendor_id(),
            props.driver_version()
        ),
        slangc_args.iter().map(|a| (*a).to_string()).collect(),
    ))
}

/// `slangc` on PATH, else `%VULKAN_SDK%\Bin\slangc`.
fn slangc() -> PathBuf {
    if let Some(sdk) = std::env::var_os("VULKAN_SDK") {
        let p = PathBuf::from(sdk).join("Bin").join(if cfg!(windows) {
            "slangc.exe"
        } else {
            "slangc"
        });
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("slangc")
}

/// `slangc` arguments that make f32 subnormals behave as IEEE: SPIR-V `DenormPreserve 32`.
/// Without them the device may flush subnormal inputs in compares and arithmetic.
pub const DENORM_PRESERVE: &[&str] = &["-denorm-mode-fp32", "preserve"];

/// True iff the SPIR-V module declares `OpCapability DenormPreserve`. Reads the words, so a
/// flag that `slangc` silently ignored cannot pass for one that took effect.
pub fn declares_denorm_preserve(spirv: &[u8]) -> bool {
    const OP_CAPABILITY: u32 = 17;
    const DENORM_PRESERVE_CAP: u32 = 4464;
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    // Header is 5 words; instructions follow, each led by (word_count << 16) | opcode.
    let mut i = 5;
    while i < words.len() {
        let (count, op) = ((words[i] >> 16) as usize, words[i] & 0xffff);
        if count == 0 {
            return false;
        }
        if op == OP_CAPABILITY && words.get(i + 1) == Some(&DENORM_PRESERVE_CAP) {
            return true;
        }
        i += count;
    }
    false
}

/// Compile Slang `source` to SPIR-V with `slangc` (plus `extra` arguments), entry point `entry`.
pub fn slang_to_spirv(
    work: &Path,
    tag: &str,
    source: &str,
    entry: &str,
    extra: &[&str],
) -> Result<Vec<u8>, String> {
    std::fs::create_dir_all(work).map_err(|e| e.to_string())?;
    let src = work.join(format!("{tag}.slang"));
    let spv = work.join(format!("{tag}.spv"));
    std::fs::write(&src, source).map_err(|e| e.to_string())?;
    let out = Command::new(slangc())
        .arg(&src)
        .args(extra)
        .args([
            "-entry", entry, "-stage", "compute", "-target", "spirv", "-o",
        ])
        .arg(&spv)
        .output()
        .map_err(|e| format!("spawn slangc: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "slangc failed:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    std::fs::read(&spv).map_err(|e| format!("read spv: {e}"))
}

/// How strictly to compare: the contract tier, the stride (1 is exhaustive), an optional waiver
/// on input bits (see [`Report::waived`]) and extra `slangc` arguments.
#[derive(Clone, Copy)]
pub struct Options {
    pub tier: Tier,
    pub stride: u32,
    pub waive: fn(u32) -> bool,
    /// Extra `slangc` arguments, e.g. [`DENORM_PRESERVE`].
    pub slangc_args: &'static [&'static str],
}

impl Options {
    /// Every f32 input, no waivers.
    pub fn exhaustive(tier: Tier) -> Self {
        Options {
            tier,
            stride: 1,
            waive: |_| false,
            slangc_args: &[],
        }
    }
}

fn ve<E: std::fmt::Display>(what: &'static str) -> impl FnOnce(E) -> String {
    move |e| format!("{what}: {e}")
}

/// Compare a Slang `cand` run on Vulkan device `device_index` (see [`vulkan_devices`]) to a cpu-c
/// `reference` over bit patterns `0, stride, 2*stride, ...`. `stride == 1` is exhaustive.
/// Mismatches on inputs where `waive(bits)` is true go to [`Report::waived`], not `mismatches`.
pub fn compare_unary_f32_vulkan(
    work: &Path,
    tag: &str,
    device_index: usize,
    reference: &Kernel,
    cand: &Kernel,
    opts: Options,
) -> Result<Report, String> {
    let spirv = slang_to_spirv(work, tag, &cand.source, &cand.name, opts.slangc_args)?;

    let inst = instance()?;
    let physical = inst
        .enumerate_physical_devices()
        .map_err(ve("enumerate"))?
        .into_iter()
        .filter(|pd| pd.find_queue_family(QueueFlags::COMPUTE).is_some())
        .nth(device_index)
        .ok_or_else(|| format!("no compute-capable Vulkan device at index {device_index}"))?;
    let qf = physical
        .find_queue_family(QueueFlags::COMPUTE)
        .ok_or("no compute queue family")?;
    let device = physical
        .create_device(DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo::single(qf)],
            ..Default::default()
        })
        .map_err(ve("create_device"))?;
    let queue = device.get_queue(qf, 0);

    let host = MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT;
    let make_buffer = || -> Result<(Buffer, DeviceMemory), String> {
        let buf = Buffer::new(
            &device,
            BufferCreateInfo {
                size: BYTES,
                usage: BufferUsage::STORAGE_BUFFER,
            },
        )
        .map_err(ve("buffer"))?;
        let req = buf.memory_requirements();
        let ty = physical
            .find_memory_type(req.memory_type_bits, host)
            .ok_or("no host-visible+coherent memory type")?;
        let mem = DeviceMemory::allocate(&device, req.size, ty).map_err(ve("allocate"))?;
        buf.bind_memory(&mem, 0).map_err(ve("bind"))?;
        Ok((buf, mem))
    };
    let (in_buf, mut in_mem) = make_buffer()?;
    let (out_buf, mut out_mem) = make_buffer()?;

    let shader = ShaderModule::from_spirv_bytes(&device, &spirv).map_err(ve("shader module"))?;
    let binding = |binding| DescriptorSetLayoutBinding {
        binding,
        descriptor_type: DescriptorType::STORAGE_BUFFER,
        descriptor_count: 1,
        stage_flags: ShaderStageFlags::COMPUTE,
    };
    let set_layout =
        DescriptorSetLayout::new(&device, &[binding(0), binding(1)]).map_err(ve("set layout"))?;
    let pool = DescriptorPool::new(
        &device,
        1,
        &[DescriptorPoolSize {
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 2,
        }],
    )
    .map_err(ve("descriptor pool"))?;
    let set = pool.allocate(&set_layout).map_err(ve("descriptor set"))?;
    set.write_buffer(0, DescriptorType::STORAGE_BUFFER, &in_buf, 0, BYTES);
    set.write_buffer(1, DescriptorType::STORAGE_BUFFER, &out_buf, 0, BYTES);
    let layout = PipelineLayout::new(&device, &[&set_layout]).map_err(ve("pipeline layout"))?;
    // slangc names the SPIR-V entry point `main` regardless of the Slang function's name.
    let pipeline =
        ComputePipeline::new(&device, &layout, &shader, "main").map_err(ve("compute pipeline"))?;

    let cmd_pool = CommandPool::new(&device, qf).map_err(ve("command pool"))?;
    let mut cmd = cmd_pool.allocate_primary().map_err(ve("command buffer"))?;
    {
        let mut rec = cmd.begin().map_err(ve("begin"))?;
        rec.bind_compute_pipeline(&pipeline);
        rec.bind_compute_descriptor_sets(&layout, 0, &[&set]);
        rec.dispatch((GPU_CHUNK as u32).div_ceil(LOCAL_SIZE), 1, 1);
        rec.memory_barrier(
            PipelineStage::COMPUTE_SHADER,
            PipelineStage::HOST,
            AccessFlags::SHADER_WRITE,
            AccessFlags::HOST_READ,
        );
        rec.end().map_err(ve("end"))?;
    }
    let fence = Fence::new(&device).map_err(ve("fence"))?;

    let run = |input: &[f32], seed: f32, out: &mut [f32]| -> Result<(), String> {
        let n = input.len();
        {
            let mut m = in_mem.map().map_err(ve("map in"))?;
            let bytes = m.as_slice_mut();
            for (dst, v) in bytes.chunks_exact_mut(4).zip(input) {
                dst.copy_from_slice(&v.to_le_bytes());
            }
        }
        {
            let mut m = out_mem.map().map_err(ve("map out"))?;
            let seed = seed.to_le_bytes();
            for dst in m.as_slice_mut().chunks_exact_mut(4) {
                dst.copy_from_slice(&seed);
            }
        }
        fence.reset().map_err(ve("fence reset"))?;
        queue.submit(&[&cmd], Some(&fence)).map_err(ve("submit"))?;
        fence.wait(u64::MAX).map_err(ve("fence wait"))?;
        let m = out_mem.map().map_err(ve("map out"))?;
        for (o, src) in out[..n].iter_mut().zip(m.as_slice().chunks_exact(4)) {
            *o = f32::from_le_bytes([src[0], src[1], src[2], src[3]]);
        }
        Ok(())
    };

    compare_with_runner(
        work,
        tag,
        reference,
        opts.tier,
        opts.stride,
        opts.waive,
        run,
    )
}
