#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <cmath>
// For MaxoutFn
#include <vector>
#include <algorithm>
#include <type_traits>

#include "Eigen/Core"

// x = Sum of all Weights * Input

// Activation Functions
// Most popular: ReLu, Sigmoid, tanh & ReLU
namespace fn_actv {

    template<typename T>
    concept FloatingTypes = std::is_same_v<float, T> || std::is_same_v<double, T> || std::is_same_v<long double, T>;

    // f(x) = x
    template <FloatingTypes T = double>
    inline T IdentityFn(const T x) {
        return x;
    };

    // System of equations
    // 0, if x < 0
    // 1, if x >= 0
    template<FloatingTypes T = double>
    inline unsigned short BinaryStepFn(const T x) {
        return x < 0 ? 0 : 1;
    };

    // f(x) = 1/(1+exp^(-x))
    // Logistic, sigmoid, or soft step
    template <FloatingTypes T = double>
    inline T SigmoidLogisticFn(const T x) {
        return 1 / (1 + std::exp(-x));
    };
    // f(x) = 1/(1+exp^(-x))
    // Logistic, sigmoid, or soft step
    template <FloatingTypes T = double>
    inline void SigmoidLogisticFn(Eigen::Vector<T, Eigen::Dynamic>& x_mtx) {
        int imax = x_mtx.rows();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < imax; ++i) {
            x_mtx(i) = 1 / (1 + std::exp(-x_mtx(i)));
        }
    };
    template void SigmoidLogisticFn<double>(Eigen::Vector<double, Eigen::Dynamic>&);

    // Hyperbolic tangent (tanh)
    // f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    template <FloatingTypes T = double>
    inline T HyperbolicTangentFn(const T x) {
        return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
    };

    // Rectified linear unit (ReLU)
    // 0        if x <= 0
    // x        if x > 0
    template <FloatingTypes T = double>
    inline T RectifiedLinearUnitFn(const T x) {
        return x > 0 ? x : 0;
    };

    // Gaussian Error Linear Unit (GELU)
    // f(x) = 1/2*x*(1+erf(x/sqrt(2))) = x * Ф(x)
    // Ф(x) = 1/2*(1+erf(x/sqrt(2)))
    template <FloatingTypes T = double>
    inline T GaussianErrorLinearUnitFn(const T x) {
        return 1.0 / 2.0 * x * (1 + std::erf(x / std::sqrt(2)));
    };

    // ln(1 + exp(x))
    template <FloatingTypes T = double>
    inline T SoftplusFn(const T x) {
        return std::log(1 + std::exp(x));
    };

    // Exponential Linear Unit (ELU)
    // a(e^x - 1)   if x <= 0
    // x            if x > 0
    template <FloatingTypes T = double>
    inline T ExponentialLinearUnitFn(const T x, const T a) {
        return x > 0 ? x : a * (std::exp(x) - 1);
    };

    // Scaled exponential linear unit (SELU) f(x)
    // lambda* alpha * (exp(x) - 1)    if x < 0
    // lambda* x                if x >= 0
    template <FloatingTypes T = double>
    inline T ScaledExponentialLinearUnitFn(const T x) {
        // lambda = 1.0507
        // alpha = 1.67326
        return (x < 0) ? 1.0507 * 1.67326 * (std::exp(x) - 1) : 1.0507 * x;
    };

    // Leaky rectified linear unit (Leaky_ReLU)
    // 0.01*x   if x < 0
    // x        if x >= 0
    template <FloatingTypes T = double>
    inline T LeakyRectifiedLinearUnitFn(const T x) {
        return x < 0 ? 0.01 * x : x;
    };

    // Parameteric rectified linear unit (PReLU)
    // a*x      if x < 0
    // x        if x >= 0
    template <FloatingTypes T = double>
    inline T ParametricRectifiedLinearUnitFn(const T x, const T a) {
        return x < 0 ? a * x : x;
    };

    // Sigmoid linear unit Sigmoid shrinkage, SiL, Swish-‍1 (SiLU)
    // x / (1 + exp(-x))
    template <FloatingTypes T = double>
    inline T SigmoidLinearUnitFn(const T x) {
        return x / (1 + std::exp(-x));
    };

    // f(x) = x*tanh(ln(1+exp^x))
    template <FloatingTypes T = double>
    inline T MishFn(const T x) {
        return x * std::tanh(std::log(1 + std::exp(x)));
    };

    // f(x) = exp^(-x^2)
    template <FloatingTypes T = double>
    inline T GaussianFn(const T x) {
        return std::exp(-std::pow(x, 2));
    };

    // f(x) = x * cos(x)
    template <FloatingTypes T = double>
    inline T GrowingCosineUnitFn(const T x) {
        return x * std::cos(x);
    };


    // Service function template for SoftmaxFn
    template <FloatingTypes T = double>
    T ExpSum(const std::vector<T>& z_vec) {
        T exp_sum{ 0 };
        for (const T& elem : z_vec) {
            exp_sum += std::exp(elem);
        }
        return exp_sum;
    }
    // f(z)[i] = exp^z[i] / Sum[j=1 to K](exp^z[j]) for i = 1,..,K; z = (z[1], ..,z[K]) in R
    // return f(z)[i] Probability of i element in vector
    // Sum[i=1 to K](f(z)[i]) = 1. Sum of all probabilities = 1
    template <FloatingTypes T = double>
    inline T SoftmaxFn(const std::vector<T>& z_vec, const size_t index) {
        return (std::exp(z_vec[index])) / (ExpSum(z_vec));
    };
    template <FloatingTypes T = double>
    inline T SoftmaxFn(const std::vector<T>& z_vec, const size_t index, const T exp_sum) {
        return std::exp(z_vec[index]) / exp_sum;
    };
    // https://en.m.wikipedia.org/wiki/Softmax_function


    // f(x) = max xi        x = Vector
    template <FloatingTypes T = double>
    inline T MaxoutFn(const std::vector<T> x) {
        return std::max_element(x.begin, x.end);
    };



    // f(x) = a*x + b
    template <FloatingTypes T = double>
    inline T LinearFn(const T x, const T a, const T b) {
        return a * x + b;
    };
    
    // f(x) = exp(-(mod(x - c)^2) / (2sigma^2))
    template <FloatingTypes T = double>
    inline T GaussianRBFFn(const T x, const T c, const T sigma) {
        return std::exp(-std::pow(x - c, 2) / (2 * std::pow(sigma, 2)));
    };

    // f(x) = 1     if a*x + b > 0
    template <FloatingTypes T = double>
    inline T HeavisideFn(const T x, const T a, const T b) {
        return (a * x + b > 0) ? 1 : 0;
    };

    // f(x) = sqrt(mod(x - c)^2 + a^2)
    template <FloatingTypes T = double>
    inline T MultiquadraticsFn(const T x, const T c, const T a) {
        return std::sqrt(std::pow(x - c, 2) + std::pow(a, 2));
    };

    // f(x) = x * (1 / (1 + exp(-x)));
    // The Swish activation function intends to be a straightforward replacement for the ubiquitous ReLU function.
    template <FloatingTypes T = double>
    inline T SwishFn(const T x) {
        return x * (1 / (1 + std::exp(-x)));
    };
    // https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/activations/swish

    // f(x) = max(0, min(1, (x + 1) / 2));
    template <FloatingTypes T = double>
    inline T HardSigmoidFn(const T x) {
        return std::fmax(0, std::fmin(1, (x + 1) / 2));
    };
    
}

#endif // !ACTIVATION_FUNCTIONS_H