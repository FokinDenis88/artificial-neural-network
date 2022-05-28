#ifndef DERIVATIVES_ACTIVATION_FUNCTIONS_HPP_
#define DERIVATIVES_ACTIVATION_FUNCTIONS_HPP_

#include <cmath>
// For MaxoutFn
#include <vector>
#include <algorithm>
#include <exception>
#include <stdexcept>


#include "Eigen/Core"

// x = Sum of all Weights * Input

// Derivatives of Activation Functions
// Most popular: ReLu, Sigmoid, tanh & ReLU
namespace fn_deriv {

    template<typename T>
    concept FloatingTypes = std::is_same_v<float, T> || std::is_same_v<double, T> || std::is_same_v<long double, T>;

    constexpr double PI{ 3.14159265358979323846 };

    // f(x)' = 1
    // f(x) = x
    inline double IdentityFn_deriv() {
        return 1.0;
    };

    // System of equations f(x)'
    // 0, if x != 0
    // undefined, if x = 0
    // 
    // System of equations f(x)
    // 0, if x < 0
    // 1, if x >= 0
    template<FloatingTypes T = double>
    inline unsigned short BinaryStepFn_deriv(const T x) {
        if (x != 0) { return 0.0; }
        else { throw std::runtime_error("If x == 0 behaviour of derivative of BinaryStepFn is undefined."); }
    };

    // f(x)' = f(x) * (1 - f(x))
    // Logistic, sigmoid, or soft step
    // f(x) = 1/(1+exp^(-x))
    template <FloatingTypes T = double>
    inline T SigmoidLogisticFn_deriv(const T f_x) {
        return f_x * (1.0 - f_x);
    };
    // f(x) = 1/(1+exp^(-x))
    // Logistic, sigmoid, or soft step
    /*template <typename T = double>
    inline void SigmoidLogisticFn(Eigen::Vector<T, Eigen::Dynamic>& x_mtx) {
        int imax = x_mtx.rows();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < imax; ++i) {
            x_mtx(i) = 1 / (1 + std::exp(-x_mtx(i)));
        }
    };*/

    // Hyperbolic tangent (tanh)
    // f(x)' = 1 - f(x)^2
    // f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    template <FloatingTypes T = double>
    inline T HyperbolicTangentFn_deriv(const T f_x) {
        return 1.0 - std::pow(f_x, 2);
    };

    // Rectified linear unit (ReLU) f(x)'
    // 0            if x < 0
    // 1            if x > 0
    // undefined    if x = 0
    //
    // Rectified linear unit (ReLU) f(x)
    // 0        if x <= 0
    // x        if x > 0
    template <FloatingTypes T = double>
    inline T RectifiedLinearUnitFn_deriv(const T x) {
        // Vanilla algorithm
        /*if (x < 0) { return 0.0; }
        if (x > 0) { return 1.0; }
        if (x == 0) { throw std::runtime_error("If x == 0 behaviour of derivative of RectifiedLinearUnitFn is undefined."); }*/
        return x <= 0 ? 0.0 : 1.0;
    };
    // https://medium.com/@kanchansarkar/relu-not-a-differentiable-function-why-used-in-gradient-based-optimization-7fef3a4cecec

    // Gaussian Error Linear Unit (GELU)
    // Ф(x) + x * ф_little(x)
    // 
    // f(x) = 1/2*x*(1+erf(x/sqrt(2))) = x * Ф(x)
    // Ф(x) = 1/2*(1+erf(x/sqrt(2)))
    template <FloatingTypes T = double>
    inline T GaussianErrorLinearUnitFn_deriv(const T F_x, const T x, const T f_x) {
        return F_x + x * f_x;
    };

    // f(x)' = 1 / (1 + exp^(-x))
    // f(x) = ln(1 + exp(x))
    template <FloatingTypes T = double>
    inline T SoftplusFn_deriv(const T x) {
        return 1.0 / (1.0 + std::exp(-x));
    };

    // Exponential Linear Unit (ELU) f'(x)
    // a*e^x        if x < 0
    // 1            if x > 0
    // 1            if x = 0 & a = 1
    // 
    // Exponential Linear Unit (ELU) f(x)
    // a(e^x - 1)   if x <= 0
    // x            if x > 0
    template <FloatingTypes T = double>
    inline T ExponentialLinearUnitFn_deriv(const T x, const T a) {
        if (x < 0) { return a * std::exp(x); }
        else if (x > 0) { return 1.0; }
        else if (x == 0 && a == 1.0) { return 1.0; }
    };

    // Scaled exponential linear unit (SELU) f(x)'
    // lambda* alpha * exp(x)       if x < 0
    // 1                            if x >= 0
    //
    // Scaled exponential linear unit (SELU) f(x)
    // lambda* alpha * (exp(x) - 1)    if x < 0
    // lambda* x                if x >= 0
    template <FloatingTypes T = double>
    inline T ScaledExponentialLinearUnitFn_deriv(const T x) {
        // lambda = 1.0507
        // alpha = 1.67326
        return (x < 0) ? 1.0507 * 1.67326 * std::exp(x) : 1.0;
    };

    // Leaky rectified linear unit (Leaky_ReLU) f(x)'
    // 0.01     if x < 0
    // 1        if x >= 0
    //
    // Leaky rectified linear unit (Leaky_ReLU) f(x)
    // 0.01*x   if x < 0
    // x        if x >= 0
    template <FloatingTypes T = double>
    inline T LeakyRectifiedLinearUnitFn_deriv(const T x) {
        return x < 0 ? 0.01 : 1.0;
    };

    // Parameteric rectified linear unit (PReLU) f(x)'
    // a        if x < 0
    // 1        if x >= 0
    //
    // Parameteric rectified linear unit (PReLU) f(x)
    // a*x      if x < 0
    // x        if x >= 0
    template <FloatingTypes T = double>
    inline T ParametricRectifiedLinearUnitFn_deriv(const T x, const T a) {
        return x < 0 ? a : 1.0;
    };

    // Sigmoid linear unit Sigmoid shrinkage, SiL, Swish-‍1 (SiLU)    f(x)'
    // (1 + exp^(-x) + x * exp^(-x)) / (1 + exp^(-x))^2
    //
    // Sigmoid linear unit Sigmoid shrinkage, SiL, Swish-‍1 (SiLU)    f(x)
    // x / (1 + exp(-x))
    template <FloatingTypes T = double>
    inline T SigmoidLinearUnitFn_deriv(const T x) {
        return (1 + std::exp(-x) + x * std::exp(-x)) / std::pow(1 + std::exp(-x), 2);
    };

    // f(x)' = very complex
    //
    // f(x) = x*tanh(ln(1+exp^x))
    template <FloatingTypes T = double>
    inline T MishFn_deriv(const T x) {
        return ( std::exp(x) * (4*std::exp(2*x) + std::exp(3*x) + 4*(1 + x) + std::exp(x)*(6 + 4*x)) ) 
                / std::pow(2 + 2 * std::exp(x) + std::exp(2 * x), 2);
    };

    // f(x)' = -2 * x * exp^(-x^2)
    //
    // f(x) = exp^(-x^2)
    template <FloatingTypes T = double>
    inline T GaussianFn_deriv(const T x) {
        return -2 * x * std::exp(-std::pow(x, 2));
    };

    // f(x)' = cos(x) - x * sin(x)
    // 
    // f(x) = x * cos(x)
    template <FloatingTypes T = double>
    inline T GrowingCosineUnitFn_deriv(const T x) {
        return std::cos(x) - x * std::sin(x);
    };


    // Deriv = S[i](vector x) * (delta[i,j] - S[j](vector x))
    // delta[i,j] = Kronecker delta
    // 0        if i != j
    // 1        if i = j
    // S - softmax vector, the result of softmax activation function
    // Deriv of softmax if matrix Jacobii of output_count * output_count
    // DjSi Matrix
    template <FloatingTypes T = double>
    inline T SoftmaxFn_deriv(const std::vector<T>& softmax_func_vec, const size_t index) {
        //return (std::exp(z_vec[index])) / (ExpSum(z_vec));
        return 0.0;
    };
    // https://en.wikipedia.org/wiki/Activation_function
    // https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    // https://www.mldawn.com/back-propagation-with-cross-entropy-and-softmax/

    // 1        if j = argmax[i](x[i])
    // 0        if j != argmax[i](x[i])
    // f(x) = max xi        x = Vector
    /*template <typename T = double>
    inline T MaxoutFn_deriv(const std::vector<T> x) {
        return std::max_element(x.begin, x.end);
    };*/


    // f(x)' = a
    //
    // f(x) = a*x + b
    template <FloatingTypes T = double>
    inline T LinearFn_deriv(const T a) {
        return a;
    };
    





    // f(x) = exp(-(mod(x - c)^2) / (2sigma^2))
    /*template <FloatingTypes T = double>
    inline T GaussianRBFFn_deriv(const T x, const T c, const T sigma) {
        return std::exp(-std::pow(x - c, 2) / (2 * std::pow(sigma, 2)));
    };*/
    // https://en.m.wikipedia.org/wiki/Radial_basis_function

    // The Dirac delta function is the derivative of the Heaviside function
    // f(x)' = 1 / (abs(a) * sqrt(PI)) * exp( -pow(x/a, 2) );
    template <FloatingTypes T = double>
    inline T HeavisideFn_deriv(const T x, const T a) {
        return 1 / (std::abs(a) * std::sqrt(PI)) * std::exp( -std::pow(x/a, 2) );
    };
    // https://en.m.wikipedia.org/wiki/Heaviside_step_function
    // https://en.m.wikipedia.org/wiki/Dirac_delta_function

    // f(x) = sqrt(mod(x - c)^2 + a^2)
    /*template <FloatingTypes T = double>
    inline T MultiquadraticsFn_deriv(const T x, const T c, const T a) {
        return std::sqrt(std::pow(x - c, 2) + std::pow(a, 2));
    };*/

    // f(x) = f_x + sigmoid * (1 - f_x)
    // sigmoid = 1 / (1 + exp(-x))
    template <FloatingTypes T = double>
    inline T SwishFn_deriv(const T f_x, const T x) {
        return f_x + (1 / (1 + std::exp(-x))) * (1 - f_x);
    };
    // https://www.google.com/search?q=Swish+function+derivative&newwindow=1&sxsrf=AOaemvKsAZFkJD7NxTRuLAZrP3nhVxgd6Q%3A1632678133728&ei=9bBQYarXK4KXrwTc2ZbYBA&oq=Swish+function+derivative&gs_lcp=ChNtb2JpbGUtZ3dzLXdpei1zZXJwEAMyBQgAEIAEMgYIABAIEB46BAgAEEc6BggAEAcQHjoECAAQDToICAAQCBAHEB46BggAEA0QHlDo7AlYvoIKYIuGCmgAcAF4AIAB4AGIAfAKkgEFMS43LjGYAQCgAQHIAQjAAQE&sclient=mobile-gws-wiz-serp

    // f(x) = max(0, min(1, (x + 1) / 2));
    /*template <FloatingTypes T = double>
    inline T HardSigmoidFn_deriv(const T x) {
        return std::max(0, std::min(1, (x + 1) / 2));
    };*/
    // https://github.com/deeplearning4j/nd4j/issues/1511
    
}

#endif // !DERIVATIVES_ACTIVATION_FUNCTIONS_HPP_