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

// Returns the number of outputs written, 0 on failure (the Rust caller
// treats 0 outputs as an error). compile() traces and evaluates the inner
// function, so Metal/GPU allocation throws can surface here.
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

  // Create C++ function wrapper
  auto cpp_fn = [fn_ptr, context](const std::vector<array>& args)
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

}  // extern "C"
