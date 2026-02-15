/* Copyright @2020-2024 Moore Threads Technology Co., Ltd("Moore Threads"). All
 * rights reserved.
 *
 * This software ("this software and its documentations" or "the software") is
 * protected by Copyright and the information contained herein is confidential.
 *
 * The software contained herein is PROPRIETARY to Moore Threads and is being
 * provided under the terms and conditions of a form of Moore Threads software
 * license agreement by and between Moore Threads and Licensee ("License
 * Agreement") or electronically accepted by Licensee. Notwithstanding any
 * terms or conditions to the contrary in the License Agreement, copy or
 * disclosure of the software to any third party without the express written
 * consent of Moore Threads is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
 * AGREEMENT, MOORE THREADS MAKES NO REPRESENTATION ABOUT ANY WARRANTIES,
 * INCLUDING BUT NOT LIMITED TO THE SUITABILITY OF THE SOFTWARE FOR ANY
 * PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF
 * ANY KIND. MOORE THREADS DISCLAIMS ALL WARRANTIES WITH REGARD TO THE
 * SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL MOORE THREADS BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THE SOFTWARE.
 */

#ifndef MUSA_MUDNN_TENSOR_H_
#define MUSA_MUDNN_TENSOR_H_

#include "mudnn_base.h"
#include <array>

namespace musa {
namespace dnn {

class Unary final : public ImplBase {
 public:
  enum class Mode {
    ADD,
    SUB,
    MUL,
    DIV,
    POW,

    SQRT,
    ROUND,
    RSQRT,
    RECIPROCAL,
    SQUARE,
    IS_FINITE,
    IS_INF,
    IS_NAN,
    IS_NONZERO,

    // BINARY_NOT,
    // LOGICAL_NOT,

    LOGICAL_AND,
    LOGICAL_OR,
    LOGICAL_XOR,

    EXP,
    // LN,
    LOG,
    LOG10,
    LOG2,
    LOG1P,

    SIN,
    COS,
    ACOS,
    TAN,
    ATAN,

    LT,
    LE,
    GT,
    GE,
    EQ,
    NE,

    MAX,
    MIN,

    CEIL,
    FLOOR,

    // nn type
    SIGMOID,
    HARDSIGMOID,
    RELU,
    LEAKY_RELU,
    RELU6,
    TANH,
    CLIP,
    ELU,
    SWISH,
    HARDSWISH,
    MISH,
    SILU,
    SOFTPLUS,
    GELU,
    GELU_TANH,

    ABS,
    ERF,

    // plain op
    IDENTITY,

    // tf
    TRUEDIV,
    FLOORDIV,
    TRUNCATEDIV,
    FLOORMOD,
    TRUNCATEMOD,

    SUB_BY_ALPHA,
    DIV_BY_ALPHA,
    TRUNCATEDIV_BY_ALPHA,
    TRUEDIV_BY_ALPHA,
    FLOORDIV_BY_ALPHA,

    CAST,
    SIGN,
  };

  Unary();
  ~Unary();

  Status SetMode(Mode m);

  Status SetAlpha(double alpha);
  Status SetAlpha(int64_t alpha);
  Status SetAlpha(const void* alpha);

  Status SetBeta(double beta);
  Status SetBeta(int64_t beta);
  Status SetBeta(const void* beta);

  Status Run(Handle& h, Tensor& out, const Tensor& in) const;
};

class Binary final : public ImplBase {
 public:
  enum class Mode {
    // math type
    ADD,
    SUB,
    MUL,
    DIV,
    POW,
    ADD_ALPHA,
    SUB_ALPHA,

    // BINARY_AND,
    // BINARY_OR,
    // BINARY_XOR,

    LOGICAL_AND,
    LOGICAL_OR,
    LOGICAL_XOR,

    // LEFT_SHIFT,
    // RIGHT_SHIFT,

    LT,
    LE,
    GT,
    GE,
    EQ,
    NE,

    MAX,
    MIN,

    // GCD,
    // LCM,

    // nn type
    PRELU,
    LEAKY_RELU_BW,
    RELU6_BW,
    THRESHOLD_BW,
    SIGMOID_BW,
    SILU_BW,
    TANH_BW,
    GELU_NONE_BW,
    GELU_TANH_BW,
    RSQRT_BW,

    TRUEDIV,
    FLOORDIV,
    TRUNCATEDIV,
    FLOORMOD,
    TRUNCATEMOD,
    DIVNONAN,
    SQUARED_DIFF,
  };

  Binary();
  ~Binary();

  Status SetMode(Mode m);

  Status SetAlpha(double alpha);
  Status SetAlpha(int64_t alpha);
  Status SetAlpha(const void* alpha);

  Status SetBeta(double beta);
  Status SetBeta(int64_t beta);
  Status SetBeta(const void* beta);

  // Follow broadcast rule as numpy, see details in:
  // https://numpy.org/doc/stable/user/basics.broadcasting.html
  Status Run(Handle& h, Tensor& out, const Tensor& l, const Tensor& r) const;
};

class Ternary final : public ImplBase {
 public:
  enum class Mode {
    // math type
    SELECT,
    SELECTV2,
    ADDCMUL,
    ADDCMUL_ALPHA,
    ADDCDIV,
    ADDCDIV_ALPHA,
    CLAMP,
  };

  Ternary();
  ~Ternary();

  Status SetMode(Mode m);

  Status SetAlpha(double alpha);
  Status SetAlpha(int64_t alpha);
  Status SetAlpha(const void* alpha);

  Status SetBeta(double beta);
  Status SetBeta(int64_t beta);
  Status SetBeta(const void* beta);

  Status SetGamma(double gamma);
  Status SetGamma(int64_t gamma);
  Status SetGamma(const void* gamma);

  Status Run(Handle& h, Tensor& out, const Tensor& in0, const Tensor& in1,
             const Tensor& in2) const;
};

class Concat final : public ImplBase {
 public:
  Concat();
  ~Concat();

  // could be negative
  Status SetAxis(int axis);

  Status Run(Handle& h, Tensor& out, int num_input, const Tensor* ins) const;
};

class Fill final : public ImplBase {
 public:
  Fill();
  ~Fill();

  Status SetValue(double value);
  Status SetValue(int64_t value);

  Status Run(Handle& h, Tensor& out) const;

  Status Run(Handle& h, Tensor& out, Tensor& mask) const;
};

class Permute final : public ImplBase {
 public:
  Permute();
  ~Permute();

  Status Run(Handle& h, Tensor& out, const Tensor& in) const;

  Status SetSrcOffset(int64_t s_offset);
  Status SetDstOffset(int64_t d_offset);

  /*
   * Permute Requirements:
   *   1. `out` and `in` must have the same dim size (i.e., shape). Generally,
   *       only dim size in `out` will be used
   *   2. these functions config the stride info in `in` and `out` to perform
   *      correct permutation
   */
  static Status ConfigDimStride(Tensor& out, Tensor& in,
                                ::std::initializer_list<int64_t> permute_dims);

  static Status ConfigDimStride(Tensor& out, Tensor& in, int len,
                                const int64_t* array_dims);
  /*
   * Slice Requirements:
   *   1. shape of `out` tensor should have been defined correctly
   *   2. the ndim of in and out tensor should be the same, as well as the
   *      length of vector start and stride
   *   3. these functions config the stride info in `in` and `out` to perform
   *      correct slice
   */
  Status ConfigDimStrideForSlice(Tensor& out, Tensor& in, const int64_t* start);

  Status ConfigDimStrideForSlice(Tensor& out, Tensor& in, const int64_t* start,
                                 const int64_t* stride);
};

class Reduce final : public ImplBase {
 public:
  enum class Mode {
    MAX,
    ADD,
    MUL,
    MEAN,
    MIN,
    PROD,
    AND,
    NORM,
    OR,
    MUL_NO_ZEROS,
    L2LOSS,
    MAX_UNSTABLE,
    MIN_UNSTABLE,
    VARIANCE,
    STD,
  };

  Reduce();
  ~Reduce();

  Status SetMode(Mode m);

  /*
   * If ndims is 0, then all dimensions are reduced, and the output tensor
   * is written into with a single element
   */
  Status SetDim(::std::initializer_list<int> dim);
  Status SetDim(int ndim, const int* dim);
  Status SetNormOrd(float ord);
  Status SetCorrection(int correction);

  Status GetWorkspaceSize(Handle& h, size_t& size_in_bytes, Tensor& out,
                          const Tensor& in);

  Status Run(Handle& h, Tensor& out, const Tensor& in,
             const MemoryMaintainer& maintainer) const;

  Status RunMeanAndVar(Handle& h, Tensor& out_var, Tensor& out_mean,
                       const Tensor& in,
                       const MemoryMaintainer& maintainer) const;

  // Only support for mode argmax/argmin, and ndim of `Reduce` must be 1
  Status RunIndices(Handle& h, Tensor& out, const Tensor& in,
                    const MemoryMaintainer& maintainer) const;

  // Only support for mode max/min, and ndim of `Reduce` must be 1
  Status RunWithIndices(Handle& h, Tensor& out, Tensor& indices,
                        const Tensor& in,
                        const MemoryMaintainer& maintainer) const;
};

}  // namespace dnn
}  // namespace musa
#endif  // MUSA_MUDNN_TENSOR_H_
