#ifndef DERIVATIVES_LOSS_FUNCTION_HPP_
#define DERIVATIVES_LOSS_FUNCTION_HPP_

#include <cmath>
#include <vector>
#include <execution>
#include <mutex>
#include <utility>
#include <algorithm>

#include "Eigen/Core"

// Loss functions
// Most popular: The mean squared error_value, cross-entropy error_value
namespace fn_loss_deriv {

    template<typename T>
    concept FloatingTypes = std::is_same_v<float, T> || std::is_same_v<double, T> || std::is_same_v<long double, T>;

    // Regression Loss Functions

    // Mean Error Loss = ME
    // dE({w[ij]}) / do[k] =  - 1 / k_max
    // Purpose: For Regression
    // Activation Function: 
    template <FloatingTypes T = double>
    inline T MeanError_deriv(const size_t k_max) {
        return -1.0 / k_max;
    };

    // Squared Error Loss = MSE
    // dE({w[ij]}) / do[k] = 2 * coeff * (output - target)
    // Purpose: For Regression
    // Activation Function: 
    template <FloatingTypes T = double>
    inline T SquaredError_deriv(const T output, const T target, const double coeff = 1) {
        return 2 * coeff * (output - target);
    };

    // Half Mean Squared Error Loss = MSE
    // half squared deriv dE / do[k] = - (t[k] - o[k])
    template <FloatingTypes T = double>
    inline T HalfSquaredError_deriv(const T output, const T target) {
        return output - target;
    };

    // Mean Squared Error Loss = MSE (L2)
    // dE / do[k] = 2 / k_max * (output - target)
    // Purpose: For Regression
    // Activation Function: 
    // MSE is sensitive towards outliers and given several examples with the same input feature values, the optimal prediction will be their mean target value. 
    template <FloatingTypes T = double>
    inline T MeanSquaredError_deriv(const T output, const T target, const size_t k_max) {
        return 2 / k_max * (output - target);
    };
    
    // Mean Squared Logarithmic Error Loss = MSLE
    // dE / o[k] = - ( 2 * (log(t[k]+1) - log(o[k]+1)) ) / (kmax * (o[k] + 1))
    // Purpose: For Regression
    // Activation Function: 
    template <FloatingTypes T = double>
    inline T MeanSquaredLogarithmicError_deriv(const T output, const T target, const size_t k_max) {
        return -(2.0 * (std::log(target + 1) - std::log(output + 1))) / (k_max * (output + 1));
    };

    // Root Mean Square Error = RMSE
    // dE / do[k] = -(t[k] - o[k]) / (k_max * |t[k] - o[k]|)
    // Purpose: For Regression
    // Activation Function: 
    template <FloatingTypes T = double>
    inline T RootMeanSquaredError_deriv(const T output, const T target, const size_t k_max) {
        return -(target - output) / (k_max * std::abs(target - output));
    };

    // Mean Absolute Error Loss = MAE (L1)
    // dE / o[k] = (1/k_max) * 1        ; o[k] > target
    // dE / o[k] = (1/k_max) * (-1)     ; o[k] < target
    // dE / o[k] = undefined            ; o[k] = target
    // Purpose: For Regression
    // Activation Function: 
    // MAE is not sensitive towards outliers and given several examples with the same input feature values, and the optimal prediction will be their median target value.
    template <FloatingTypes T = double>
    inline T MeanAbsoluteError_deriv(const T output, const T target, const size_t k_max) {
        return output > target ? 1.0 / k_max : -1.0 / k_max;
    };
    // https://stats.stackexchange.com/questions/312737/mean-absolute-error-mae-derivative

    // Poison
    // dE / do[k] = 1/k_max * (1 - target[k] / output[k])
    // Purpose: For Regression
    // Activation Function: 
    // Use the Poisson loss when you believe that the target value comes from a Poisson distribution and want to model the rate parameter conditioned on some input.
    template <FloatingTypes T = double>
    inline T Poison_deriv(const T output, const T target, const size_t k_max) {
        return 1.0 / k_max * (1.0 - target / output);
    };
    // https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/poisson

// !Regression Loss Functions

// Binary Classification Loss Functions

    // Binary Crossentropy or Logs Loss (Maybe Sparse_Multiclass_Cross_Entropy also?)
    // dE / do[k] = - 1/k_max * (t[k] / o[k] - (1 - t[k]) / (1 - o[k]))
    // Purpose: For Binary Classification
    // Activation Function: 
    template <FloatingTypes T = double>
    inline T BinaryCrossentropy_deriv(const T output, const T target, const size_t k_max) {
        return -1.0 / k_max * (target / output - (1.0 - target) / (1.0 - output));
    };
    // https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right

    // Hinge Loss
    // E({w[ij]}) = (1/k_max) * Sum[k in output, k_max](max(0, 1 - y_target * y_output))
    // Purpose: For Binary Classification
    // Activation Function: 
    /*template <FloatingTypes T = double>
    inline T HingeLoss_deriv(const std::vector<T>& output, const std::vector<T>& target) {
        const int output_size = output.size();
        T loss{ 0 };
        for (int i = 0; i < output_size; ++i) {
            loss += std::max<T>(0, 1 - target[i] * output[i]);
        }
        return loss;
    };*/

    // Squared hinge
    // E({w[ij]}) = (1/k_max) * Sum[k in output, k_max](max(0, 1 - y_target * y_output))^2
    // Purpose: For Binary Classification
    // Activation Function: tanh() activation function in the last layer 
    // The squared hinge loss is a loss function used for “maximum margin” binary classification problems.
    // Use the Squared Hinge loss function on problems involving yes/no (binary) decisions and when you’re not interested in knowing how certain the classifier is about the classification (i.e., when you don’t care about the classification probabilities). Use in combination with the tanh() activation function in the last layer.
    /*template <FloatingTypes T = double>
    inline T SquaredHinge_deriv(const std::vector<T>& output, const std::vector<T>& target) {
        const int output_size = output.size();
        T loss{ 0 };
        for (int i = 0; i < output_size; ++i) {
            loss += std::pow(std::max<T>(0, 1 - target[i] * output[i]), 2);
        }
        return loss;
    };*/

    // !Binary Classification Loss Functions

    // Multi-Class Single Label Classification Loss Functions
    // Target value is label. One class can have more than one label target value.

    // Categorical Crossentropy
    // E({w[ij]}) = -(1/k_max) * Sum[k in output, k_max](y_target * log(y_output))
    // Purpose: For Multi-Class Classification
    // Activation Function: sigmoid, tanh
    // This loss is a very good measure of how distinguishable two discrete probability distributions are from each other. 
    /*template <FloatingTypes T = double>
    inline T CategoricalCrossentropy_deriv(const std::vector<T>& output, const std::vector<T>& target) {
        const int output_size = output.size();
        T loss{ 0 };
        for (int i = 0; i < output_size; ++i) {
            loss += target[i] * std::log(output[i]);
        }
        loss *= -1;
        return loss;
    };*/
    // https://www.mldawn.com/back-propagation-with-cross-entropy-and-softmax/
    // https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    // https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    // https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy


    // Kullback-Liebler Divergence LOSS (KL-Divergence)
    // D[kl](P||Q) = Sum[k in output, k_max]( P(x) * ln(P(x) / Q(x)) )
    // Purpose: For Multi-Class Classification
    //  The goal of the KL divergence loss is to approximate the true probability distribution P of our target variables with respect to the input features, given some approximate distribution Q. 
    // This Can be achieved by minimizing the Dkl(P||Q) then it is called forward KL.
    // If we are minimizing Dkl(Q||P) then it is called backward KL.
    // Forward KL → applied in Supervised Learning
    // Backward KL → applied in Reinforcement learning
    /*template <FloatingTypes T = double>
    inline T KullbackLieblerDivergenceForward_deriv(const std::vector<T>& output, const std::vector<T>& target) {
        T loss{ 0 };
        for (int i = 0; i < output_size; ++i) {
            loss += target[i] * std::log(output[i]);
        }
        return loss;
    };*/
    // https://medium.com/@zeeshanmulla/cost-activation-loss-function-neural-network-deep-learning-what-are-these-91167825a4de

    // Kullback-Liebler Divergence LOSS (KL-Divergence)
    // D[kl](P||Q) = Sum[k in output, k_max]( P(x) * ln(P(x) / Q(x)) )
    // Purpose: For Multi-Class Classification
    //  The goal of the KL divergence loss is to approximate the true probability distribution P of our target variables with respect to the input features, given some approximate distribution Q. 
    // This Can be achieved by minimizing the Dkl(P||Q) then it is called forward KL.
    // If we are minimizing Dkl(Q||P) then it is called backward KL.
    // Forward KL → applied in Supervised Learning
    // Backward KL → applied in Reinforcement learning
    /*template <FloatingTypes T = double>
    inline T KullbackLieblerDivergenceBackward_deriv(const std::vector<T>& output, const std::vector<T>& target) {
        const int output_size = output.size();
        T loss{ 0 };
        for (int i = 0; i < output_size; ++i) {
            loss += target[i] * std::log(output[i]);
        }
        loss *= -1;
        return loss;
    };*/


    // Focal Loss
    // E({w[ij]}) = -(1/k_max) * Sum[k in output, k_max]((1 - output[i])^gamma * target[i] * log(output[i]))
    // (1 - output[i])^gamma is the weighting factor
    // Purpose: For Multi-Class Classification
    // Activation Function: 
    // Use the focal loss function in single-label classification tasks as an alternative to the more commonly used categorical crossentropy.
    /*template <FloatingTypes T = double>
    inline T FocalLoss_deriv(const std::vector<T>& output, const std::vector<T>& target, const double gamma) {
        const int output_size = output.size();
        T loss{ 0 };
        for (int i = 0; i < output_size; ++i) {
            loss += std::pow(1 - output[i], gamma) * target[i] * std::log(output[i]);
        }
        return loss;
    };*/
    // https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/focal-loss


    // Focal Loss
    // 1/2 * a^2                    ; for |a| <= delta
    // delta * (|a| - 1/2*delata)     otherwise
    // Purpose: For Multi-Class Classification
    // Activation Function: 
    // The Huber loss function describes the penalty incurred by an estimation procedure f. 
    /*template <FloatingTypes T = double>
    inline T HuberLoss_deriv(const std::vector<T>& output, const std::vector<T>& target, const double delta) {
        const int output_size = output.size();
        T loss{ 0 };
        T a{};
        for (int i = 0; i < output_size; ++i) {
            a = target[i] - output[i];
            if (std::abs(a) <= delta) { loss = std::pow(a, 2) / 2; }
            else { loss = delta * std::abs(a) - std::pow(delta, 2) / 2; }
        }
        return loss;
    };*/
    // https://en.wikipedia.org/wiki/Huber_loss

// !Multi-Class Single Label Classification Loss Functions

// Multi-Class Multi Label Classification Loss Functions
// !Multi-Class Multi Label Classification Loss Functions
    
}

#endif // !DERIVATIVES_LOSS_FUNCTION_HPP_