// mlx_transforms.cpp — vmap / compile FFI shims for genmlx-core.
//
// Ported from the robert-johansson/mlx-node fork (MLX 0.31.2 @49503b65) and
// reconciled against OUR frozen MLX 0.32.0 @b410f6c headers
// (mlx/transforms.h, mlx/compile.h).
//
// Signature reconciliation (0.32.0):
//   mlx::core::vmap(
//     const std::function<std::vector<array>(const std::vector<array>&)>& fun,
//     const std::vector<int>& in_axes = {},
//     const std::vector<int>& out_axes = {})           // transforms.h:184-187
//   mlx::core::compile(
//     std::function<std::vector<array>(const std::vector<array>&)> fun,
//     bool shapeless = false)                            // compile.h:13-15
// Both are byte-identical to the API the fork shims targeted, so the shim
// bodies port verbatim — no drift. Bodies wrapped in MLX_GUARD_* per P1.
//
// mlx_common.h supplies: #include "mlx/transforms.h", #include "mlx/compile.h",
// `using mlx::core::array;`, `struct mlx_array;`, and the guard macros.

#include "mlx_common.h"

#include "mlx/backend/gpu/replay_capture.h" // genmlx-7prh captured replay
#include "mlx/primitives.h" // full Primitive type for the tape walk's stream check

extern "C" {

// ============================================================================
// Vmap: vectorize a function over arrays
//
// Takes a C function pointer callback (called synchronously) and returns
// the vmapped results.
// ============================================================================

typedef mlx_array* (*VmapFunctionPtr)(mlx_array* const* inputs,
                                       size_t input_count,
                                       void* context);

mlx_array* mlx_vmap_apply(VmapFunctionPtr fn_ptr,
                           void* context,
                           mlx_array* const* input_handles,
                           size_t input_count,
                           const int32_t* in_axes, size_t in_axes_len,
                           const int32_t* out_axes, size_t out_axes_len,
                           mlx_array** output_handles,
                           size_t max_outputs,
                           size_t* num_outputs) {
  MLX_GUARD_PTR("vmap_apply",
  // Convert input handles to arrays
  std::vector<array> inputs;
  inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; i++) {
    inputs.emplace_back(*reinterpret_cast<array*>(input_handles[i]));
  }

  // Convert axes
  std::vector<int> in_ax(in_axes, in_axes + in_axes_len);
  std::vector<int> out_ax(out_axes, out_axes + out_axes_len);

  // Create the C++ function wrapper
  auto cpp_fn = [fn_ptr, context](const std::vector<array>& args)
      -> std::vector<array> {
    // Convert to handles
    std::vector<mlx_array*> handles;
    handles.reserve(args.size());
    for (const auto& arr : args) {
      handles.push_back(reinterpret_cast<mlx_array*>(new array(arr)));
    }

    // Call the user function
    mlx_array* result = fn_ptr(handles.data(), handles.size(), context);

    // Clean up input handles
    for (auto* h : handles) {
      delete reinterpret_cast<array*>(h);
    }

    if (!result) return {};
    auto* arr = reinterpret_cast<array*>(result);
    std::vector<array> out = {std::move(*arr)};
    delete arr;
    return out;
  };

  // Apply vmap
  auto vmapped = mlx::core::vmap(cpp_fn, in_ax, out_ax);
  auto results = vmapped(inputs);

  // Store outputs
  *num_outputs = std::min(results.size(), max_outputs);
  for (size_t i = 0; i < *num_outputs; i++) {
    output_handles[i] =
        reinterpret_cast<mlx_array*>(new array(std::move(results[i])));
  }

  return nullptr; // success
  )
}

// ============================================================================
// Compile: JIT-compile a function
// ============================================================================

typedef size_t (*CompileFunctionPtr)(mlx_array* const* inputs,
                                      size_t input_count,
                                      mlx_array** outputs,
                                      size_t max_outputs,
                                      void* context);

// The trace trampoline: bridges mlx::core::compile's trace calls back to the
// caller's callback (Rust -> JS). Shared by the one-shot mlx_compile_apply
// and the persistent mlx_compile_create below. The (fn_ptr, context) pair
// must stay valid for as long as the returned std::function may trace —
// one call for the one-shot path, the handle's whole lifetime for the
// persistent path (shape-change retraces call back in).
static std::function<std::vector<array>(const std::vector<array>&)>
make_compile_trampoline(CompileFunctionPtr fn_ptr, void* context) {
  return [fn_ptr, context](const std::vector<array>& args)
      -> std::vector<array> {
    std::vector<mlx_array*> handles;
    handles.reserve(args.size());
    for (const auto& arr : args) {
      handles.push_back(reinterpret_cast<mlx_array*>(new array(arr)));
    }

    constexpr size_t MAX_OUT = 16;
    mlx_array* out_handles[MAX_OUT] = {};
    size_t num_out = fn_ptr(handles.data(), handles.size(),
                             out_handles, MAX_OUT, context);

    for (auto* h : handles) {
      delete reinterpret_cast<array*>(h);
    }

    std::vector<array> outputs;
    outputs.reserve(num_out);
    for (size_t i = 0; i < num_out; i++) {
      auto* arr = reinterpret_cast<array*>(out_handles[i]);
      outputs.push_back(std::move(*arr));
      delete arr;
    }
    return outputs;
  };
}

// Returns the number of outputs written, 0 on failure (the Rust caller
// treats 0 outputs as an error). compile() traces and evaluates the inner
// function, so Metal/GPU allocation throws can surface here.
//
// NOTE (genmlx-z2gt): this one-shot form creates a FRESH closure identity per
// call, so mlx::core::compile re-traces every invocation — it cannot amortize.
// For trace-once/replay-many use mlx_compile_create/mlx_compiled_call below.
size_t mlx_compile_apply(CompileFunctionPtr fn_ptr,
                          void* context,
                          mlx_array* const* input_handles,
                          size_t input_count,
                          bool shapeless,
                          mlx_array** output_handles,
                          size_t max_outputs) {
  MLX_GUARD_VAL("compile_apply", 0,
  // Convert input handles
  std::vector<array> inputs;
  inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; i++) {
    inputs.emplace_back(*reinterpret_cast<array*>(input_handles[i]));
  }

  auto cpp_fn = make_compile_trampoline(fn_ptr, context);

  // Compile and apply
  auto compiled = mlx::core::compile(cpp_fn, shapeless);
  auto results = compiled(inputs);

  size_t count = std::min(results.size(), max_outputs);
  for (size_t i = 0; i < count; i++) {
    output_handles[i] =
        reinterpret_cast<mlx_array*>(new array(std::move(results[i])));
  }
  return count;
  )
}

// ============================================================================
// Persistent compile (genmlx-z2gt Phase 1): trace once, replay in C++.
//
// mlx::core::compile keys its trace cache by closure identity. Holding the
// RETURNED compiled closure in a heap handle keeps one stable identity for
// the handle's lifetime: the first call per input-shape traces (invoking the
// trampoline back into the caller's runtime); every other call replays the
// cached graph without touching the callback. The (fn_ptr, context) pair
// must therefore outlive the handle — freeing the context before
// mlx_compiled_free is use-after-free on the next shape-change retrace.
// ============================================================================

struct mlx_compiled_fn {
  std::function<std::vector<array>(const std::vector<array>&)> fn;
  // ---- Replay capture (genmlx-7prh) ---------------------------------------
  // After a successful capture, calls through mlx_compiled_call_captured are
  // launch-only: memcpy the inputs into the staged buffers, launch the
  // retained execs, sync, copy the outputs out. capture_tried latches so a
  // failed/unsupported capture falls back to the plain replay path forever
  // (no per-call re-probing).
  bool capture_tried{false};
  bool captured{false};
  void* sink{nullptr}; // gpu::replay_capture_* handle
  std::vector<array> staged_inputs; // owned buffers; per-call memcpy targets
  std::vector<array> retained_tape; // every array of the captured graph
  std::vector<array> retained_outputs; // buffers the launches write into
};

// Iterative walk of the lazy graph reachable from `outs` (pre-eval, links
// intact): collect every array (buffers must outlive the capture) and reject
// tapes with any non-GPU op — a CPU-stream primitive would run outside the
// captured graphs. Iterative on purpose: MALA-class chains reach 30k+ nodes
// and a recursive walk would risk the host stack (cf. genmlx-a1uh).
static bool collect_capture_tape(
    const std::vector<array>& outs,
    std::vector<array>& tape) {
  std::unordered_map<std::uintptr_t, bool> visited;
  std::vector<array> stack(outs.begin(), outs.end());
  while (!stack.empty()) {
    array a = std::move(stack.back());
    stack.pop_back();
    if (!visited.emplace(a.id(), true).second) {
      continue;
    }
    if (a.has_primitive() &&
        a.primitive().stream().device != mlx::core::Device::gpu) {
      return false;
    }
    for (auto& s : a.siblings()) {
      stack.push_back(s);
    }
    for (auto& in : a.inputs()) {
      stack.push_back(in);
    }
    tape.push_back(std::move(a));
  }
  return true;
}

// Returns an opaque handle, or nullptr on failure. Destroying the handle
// (mlx_compiled_free) destroys the compiled closure, which releases the
// cached trace via MLX's own compile-cache cleanup.
void* mlx_compile_create(CompileFunctionPtr fn_ptr,
                         void* context,
                         bool shapeless) {
  MLX_GUARD_VAL("compile_create", nullptr,
  auto* handle = new mlx_compiled_fn{
      mlx::core::compile(make_compile_trampoline(fn_ptr, context), shapeless)};
  return reinterpret_cast<void*>(handle);
  )
}

size_t mlx_compiled_call(void* handle,
                         mlx_array* const* input_handles,
                         size_t input_count,
                         mlx_array** output_handles,
                         size_t max_outputs) {
  MLX_GUARD_VAL("compiled_call", 0,
  auto* h = reinterpret_cast<mlx_compiled_fn*>(handle);
  std::vector<array> inputs;
  inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; i++) {
    inputs.emplace_back(*reinterpret_cast<array*>(input_handles[i]));
  }

  auto results = h->fn(inputs);

  size_t count = std::min(results.size(), max_outputs);
  for (size_t i = 0; i < count; i++) {
    output_handles[i] =
        reinterpret_cast<mlx_array*>(new array(std::move(results[i])));
  }
  return count;
  )
}

// Captured-replay call (genmlx-7prh). Unlike mlx_compiled_call this returns
// EVALUATED outputs (fresh arrays — never aliases the retained buffers):
// the eval is part of the call so the capture window can wrap it, and on the
// launch-only path there is nothing left to evaluate. First call per handle
// attempts the capture; any failure (unsupported backend, CPU op in the
// tape, graph fallback, foreign-thread commit) latches a permanent fallback
// to trace-cache replay + eval — behaviorally identical, just slower.
size_t mlx_compiled_call_captured(void* handle,
                                  mlx_array* const* input_handles,
                                  size_t input_count,
                                  mlx_array** output_handles,
                                  size_t max_outputs) {
  MLX_GUARD_VAL("compiled_call_captured", 0,
  namespace gpu = mlx::core::gpu;
  auto* h = reinterpret_cast<mlx_compiled_fn*>(handle);
  std::vector<array> inputs;
  inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; i++) {
    inputs.emplace_back(*reinterpret_cast<array*>(input_handles[i]));
  }

  auto emit = [&](std::vector<array>&& results) -> size_t {
    size_t count = std::min(results.size(), max_outputs);
    for (size_t i = 0; i < count; i++) {
      output_handles[i] =
          reinterpret_cast<mlx_array*>(new array(std::move(results[i])));
    }
    return count;
  };
  auto plain = [&]() -> size_t {
    auto results = h->fn(inputs);
    mlx::core::eval(results);
    return emit(std::move(results));
  };

  if (h->captured) {
    // Launch-only replay. Shape/dtype drift falls back to the trace-cache
    // path (mlx::core::compile keeps per-shape traces); the capture stays
    // bound to the shape it was taken at.
    if (inputs.size() != h->staged_inputs.size()) {
      return plain();
    }
    for (size_t i = 0; i < inputs.size(); i++) {
      if (inputs[i].shape() != h->staged_inputs[i].shape() ||
          inputs[i].dtype() != h->staged_inputs[i].dtype()) {
        return plain();
      }
    }
    mlx::core::eval(inputs); // memcpy sources must be complete
    for (size_t i = 0; i < inputs.size(); i++) {
      if (!gpu::replay_capture_copy_into(
              h->sink, h->staged_inputs[i], inputs[i])) {
        return plain();
      }
    }
    gpu::replay_capture_launch(h->sink);
    std::vector<array> outs;
    outs.reserve(h->retained_outputs.size());
    for (auto& ro : h->retained_outputs) {
      auto c = gpu::replay_capture_clone_array(h->sink, ro);
      if (!c) {
        return plain();
      }
      outs.push_back(std::move(*c));
    }
    gpu::replay_capture_sync(h->sink);
    return emit(std::move(outs));
  }

  if (h->capture_tried) {
    return plain();
  }
  h->capture_tried = true;

  // ---- Capture attempt ----------------------------------------------------
  mlx::core::eval(inputs); // staging clones read these buffers
  void* sink = gpu::replay_capture_begin();
  if (!sink) {
    return plain();
  }
  std::vector<array> staged;
  staged.reserve(inputs.size());
  for (auto& in : inputs) {
    auto c = gpu::replay_capture_clone_array(sink, in);
    if (!c) {
      gpu::replay_capture_free(sink);
      return plain();
    }
    staged.push_back(std::move(*c));
  }
  gpu::replay_capture_sync(sink); // staged data ready before the eval reads it

  std::vector<array> outs;
  std::vector<array> tape;
  bool tape_ok = false;
  try {
    outs = h->fn(staged); // first call traces via the JS builder
    tape_ok = collect_capture_tape(outs, tape);
    if (tape_ok) {
      mlx::core::eval(outs); // the captured eval
    }
  } catch (...) {
    // The window MUST close before rethrowing — a live global sink would
    // capture every later eval in the process.
    gpu::replay_capture_end(sink, nullptr);
    gpu::replay_capture_free(sink);
    throw;
  }
  if (!tape_ok) {
    gpu::replay_capture_end(sink, nullptr);
    gpu::replay_capture_free(sink);
    return plain(); // re-run against the ORIGINAL inputs, uncaptured
  }
  std::string why;
  bool ok = gpu::replay_capture_end(sink, &why);
  if (!ok || gpu::replay_capture_graph_count(sink) == 0) {
    gpu::replay_capture_free(sink);
    // outs are correct and evaluated — only the capture failed.
    return emit(std::move(outs));
  }

  // Return copies: the retained output buffers are the next launch's
  // destination, so the capture call's results must not alias them.
  std::vector<array> copies;
  copies.reserve(outs.size());
  for (auto& o : outs) {
    auto c = gpu::replay_capture_clone_array(sink, o);
    if (!c) {
      gpu::replay_capture_free(sink);
      return emit(std::move(outs));
    }
    copies.push_back(std::move(*c));
  }
  gpu::replay_capture_sync(sink);

  h->sink = sink;
  h->staged_inputs = std::move(staged);
  h->retained_tape = std::move(tape);
  h->retained_outputs = std::move(outs);
  h->captured = true;
  return emit(std::move(copies));
  )
}

// True when the handle completed a capture and replays are launch-only —
// the honesty probe for tests and the CLJS layer (no silent path ambiguity).
bool mlx_compiled_is_captured(void* handle) {
  auto* h = reinterpret_cast<mlx_compiled_fn*>(handle);
  return h != nullptr && h->captured;
}

void mlx_compiled_free(void* handle) {
  auto* h = reinterpret_cast<mlx_compiled_fn*>(handle);
  if (h && h->sink) {
    mlx::core::gpu::replay_capture_free(h->sink);
  }
  delete h;
}

}  // extern "C"
