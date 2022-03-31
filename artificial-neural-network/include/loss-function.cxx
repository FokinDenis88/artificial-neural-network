#ifndef LOSS_FUNCTION_HPP_
#define LOSS_FUNCTION_HPP_

module;

#include <cmath>
#include <vector>
#include <execution>
#include <mutex>
#include <utility>
#include <algorithm>

#include "Eigen/Core"

export module loss_function;

// Loss functions
// Most popular: The mean squared error_value, cross-entropy error_value
export namespace fn_loss {
    // https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-squared-logarithmic-loss-(msle)
    // https://medium.com/@zeeshanmulla/cost-activation-loss-function-neural-network-deep-learning-what-are-these-91167825a4de

    template<typename T>
    concept FloatingTypes = std::is_same_v<float, T> || std::is_same_v<double, T> || std::is_same_v<long double, T>;

// Regression Loss Functions

    // Mean Error Loss = ME
    // E({w[ij]}) = (1/k_max) * Sum[k in output, k_max]|t[k]-o[k]|
    // Purpose: For Regression
    // Activation Function: 
    template <FloatingTypes T = double>
    inline T MeanError(const std::vector<T>& output, const std::vector<T>& target) {
        T loss{ 0 };
        int output_size = output.size();
        for (int i = 0; i < output_size; ++i) {
            loss += target[i] - output[i];
        }
        loss /= output_size;
        return loss;
    };

    // Squared Error Loss = MSE
    // E({w[ij]}) = (1/k_max) * Sum[k in output, k_max]((t[k]-o[k])^2)
    // Purpose: For Regression
    // Activation Function: 
    template <FloatingTypes T = double>
    inline T SquaredError(const std::vector<T>& output, const std::vector<T>& target, const double coeff = 1) {
        const int output_size = output.size();
        T loss{ 0 };
        for (int i = 0; i < output_size; ++i) {
            loss += std::pow(target[i] - output[i], 2);
        }
        loss *= coeff;
        return loss;
    };

    // Half Mean Squared Error Loss = MSE
    // E({w[ij]}) = (1/2) * Sum[k in output, k_max]((t[k]-o[k])^2)
    // Purpose: For Regression
    // Activation Function: 
    // Half of Mean Square will increase computational performance, because of less mathematical operations
    template <FloatingTypes T = double>
    inline T HalfSquaredError(const std::vector<T>& output, const std::vector<T>& target, const double coeff = 1) {
        const int output_size = output.size();
        T loss{ 0 };
        for (int i = 0; i < output_size; ++i) {
            loss += std::pow(target[i] - output[i], 2);
        }
        loss /= 2;
        return loss;
    };

    // Mean Squared Error Loss = MSE (L2)
    // E({w[ij]}) = (1/k_max) * Sum[k in output, k_max]((t[k]-o[k])^2)
    // Purpose: For Regression
    // Activation Function: 
    // MSE is sensitive towards outliers and given several examples with the same input feature values, the optimal prediction will be their mean target value. 
    template <FloatingTypes T = double>
    inline T MeanSquaredError(const std::vector<T>& output, const std::vector<T>& target) {
        const int output_size = output.size();
        T loss{ 0 };
        for (int i = 0; i < output_size; ++i) {
            loss += std::pow(target[i] - output[i], 2);
        }
        loss /= output_size;
        return loss;
    };
    // Mean Squared Error Loss = MSE
    // E({w[ij]}) = (1/k_max) * Sum[k in output, k_max]((t[k]-o[k])^2)
    // MSE = 1/n * Sum(i=1 to n){(Y[output] - Y[target])^2}
    // MSE = 1/n * Error[transpont] * Error
    // Error = Y[output] - Y[target]
    // Purpose: For Regression
    inline Eigen::VectorXd MeanSquaredError(const Eigen::VectorXd& output_vec, const Eigen::VectorXd& target_vec) {
        const size_t n = output_vec.size();
                         // Error = Y[output] - Y[target]
        Eigen::VectorXd loss_vec{ target_vec - output_vec };

              // MSE = 1/n * Sum(i=1 to n){(Y[output] - Y[target])^2} 
                  // MSE = 1/n * Error[transpont] * Error
        return 1.0 / n * loss_vec.transpose() * loss_vec;
    };

    // Mean Squared Logarithmic Error Loss = MSLE
    // E({w[ij]}) = (1/k_max) * Sum[k in output, k_max]( log(t[k] + 1) - log(o[k] + 1)) )^2
    // Purpose: For Regression
    // Activation Function: 
    template <FloatingTypes T = double>
    inline T MeanSquaredLogarithmicError(const std::vector<T>& output, const std::vector<T>& target) {
        const int output_size = output.size();
        T loss{ 0 };
        for (int i = 0; i < output_size; ++i) {
            loss += std::pow(std::log(target[i] + 1) - std::log(output[i] + 1), 2);
        }
        loss /= output_size;
        return loss;
    };

    // Root Mean Square Error = RMSE
    // E({w[ij]}) = sqrt(MSE)
    // Purpose: For Regression
    // Activation Function: 
    template <FloatingTypes T = double>
    inline T RootMeanSquaredError(const std::vector<T>& output, const std::vector<T>& target) {
        return std::sqrt(MeanSquaredError(output, target));
    };

    // Mean Absolute Error Loss = MAE (L1)
    // E({w[ij]}) = (1/k_max) * Sum[k in output, k_max]|y_output - y_target|
    // Purpose: For Regression
    // Activation Function: 
    // MAE is not sensitive towards outliers and given several examples with the same input feature values, and the optimal prediction will be their median target value.
    template <FloatingTypes T = double>
    inline T MeanAbsoluteError(const std::vector<T>& output, const std::vector<T>& target) {
        const int output_size = output.size();
        T loss{ 0 };
        for (int i = 0; i < output_size; ++i) {
            loss += std::abs(output[i] - target[i]);
        }
        loss /= output_size;
        return loss;
    };

    // Poison
    // E({w[ij]}) = (1/k_max) * Sum[k in output, k_max](y_output - y_target * log(y_output))
    // Purpose: For Regression
    // Activation Function: 
    // Use the Poisson loss when you believe that the target value comes from a Poisson distribution and want to model the rate parameter conditioned on some input.
    template <FloatingTypes T = double>
    inline T Poison(const std::vector<T>& output, const std::vector<T>& target) {
        const int output_size = output.size();
        T loss{ 0 };
        for (int i = 0; i < output_size; ++i) {
            loss += output[i] - target[i] * std::log(output[i]);
        }
        loss /= output_size;
        return loss;
    };
    // https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/poisson
    
// !Regression Loss Functions

// Binary Classification Loss Functions

    // Binary Crossentropy or Logs Loss (Maybe Sparse_Multiclass_Cross_Entropy also?)
    // E({w[ij]}) = -(1/k_max) * Sum[k in output, k_max](y_target * log(y_predict) + (1 - y_target) * log(1 - y_predict))
    // Purpose: For Binary Classification
    // Activation Function: 
    template <FloatingTypes T = double>
    inline T BinaryCrossentropy(const std::vector<T>& output, const std::vector<T>& target) {
        const int output_size = output.size();
        T loss{ 0 };
        for (int i = 0; i < output_size; ++i) {
            loss += target[i] * std::log(output[i]) + (1 - target[i]) * std::log(1 - output[i]);
        }
        loss /= -output_size;
        return loss;
    };
    // https://www.analyticsvidhya.com/blog/2021/03/binary-cross-entropy-log-loss-for-binary-classification/#:~:text=What%20is%20Binary%20Cross%20Entropy,far%20from%20the%20actual%20value.
    // https://datawookie.dev/blog/2015/12/making-sense-of-logarithmic-loss/#:~:text=Logarithmic%20Loss%2C%20or%20simply%20Log,evaluation%20metric%20in%20Kaggle%20competitions.&text=Log%20Loss%20quantifies%20the%20accuracy%20of%20a%20classifier%20by%20penalising%20false%20classifications.

    // Hinge Loss
    // E({w[ij]}) = (1/k_max) * Sum[k in output, k_max](max(0, 1 - y_target * y_output))
    // Purpose: For Binary Classification
    // Activation Function: 
    template <FloatingTypes T = double>
    inline T HingeLoss(const std::vector<T>& output, const std::vector<T>& target) {
        const int output_size = output.size();
        T loss{ 0 };
        for (int i = 0; i < output_size; ++i) {
            loss += std::max<T>(0, 1 - target[i] * output[i]);
        }
        return loss;
    };

    // Squared hinge
    // E({w[ij]}) = (1/k_max) * Sum[k in output, k_max](max(0, 1 - y_target * y_output))^2
    // Purpose: For Binary Classification
    // Activation Function: tanh() activation function in the last layer 
    // The squared hinge loss is a loss function used for “maximum margin” binary classification problems.
    // Use the Squared Hinge loss function on problems involving yes/no (binary) decisions and when you’re not interested in knowing how certain the classifier is about the classification (i.e., when you don’t care about the classification probabilities). Use in combination with the tanh() activation function in the last layer.
    template <FloatingTypes T = double>
    inline T SquaredHinge(const std::vector<T>& output, const std::vector<T>& target) {
        const int output_size = output.size();
        T loss{ 0 };
        for (int i = 0; i < output_size; ++i) {
            loss += std::pow(std::max<T>(0, 1 - target[i] * output[i]), 2);
        }
        return loss;
    };

// !Binary Classification Loss Functions

// Multi-Class Single Label Classification Loss Functions
// Target value is label. One class can have more than one label target value.

    // Categorical Crossentropy
    // E({w[ij]}) = -(1/k_max) * Sum[k in output, k_max](y_target * log(y_output))
    // Purpose: For Multi-Class Classification
    // Activation Function: sigmoid, tanh
    // This loss is a very good measure of how distinguishable two discrete probability distributions are from each other. 
    template <FloatingTypes T = double>
    inline T CategoricalCrossentropy(const std::vector<T>& output, const std::vector<T>& target) {
        const int output_size = output.size();
        T loss{ 0 };
        for (int i = 0; i < output_size; ++i) {
            loss += target[i] * std::log(output[i]);
        }
        loss *= -1;
        return loss;
    };
    // https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/categorical-crossentropy
    // https://medium.com/@zeeshanmulla/cost-activation-loss-function-neural-network-deep-learning-what-are-these-91167825a4de


    // Kullback-Liebler Divergence LOSS (KL-Divergence)
    // D[kl](P||Q) = Sum[k in output, k_max]( P(x) * ln(P(x) / Q(x)) )
    // Purpose: For Multi-Class Classification
    //  The goal of the KL divergence loss is to approximate the true probability distribution P of our target variables with respect to the input features, given some approximate distribution Q. 
    // This Can be achieved by minimizing the Dkl(P||Q) then it is called forward KL.
    // If we are minimizing Dkl(Q||P) then it is called backward KL.
    // Forward KL → applied in Supervised Learning
    // Backward KL → applied in Reinforcement learning
    template <FloatingTypes T = double>
    inline T KullbackLieblerDivergenceForward(const std::vector<T>& output, const std::vector<T>& target) {
        T loss = 0.0;
        for (int i = 0, imax = output.size(); i < imax; ++i) {
            loss += target[i] * std::log(target[i] / output[i]);
        }
        return loss;
    };
    // https://medium.com/@zeeshanmulla/cost-activation-loss-function-neural-network-deep-learning-what-are-these-91167825a4de

    // Kullback-Liebler Divergence LOSS (KL-Divergence)
    // D[kl](P||Q) = Sum[k in output, k_max]( P(x) * ln(P(x) / Q(x)) )
    // Purpose: For Multi-Class Classification
    //  The goal of the KL divergence loss is to approximate the true probability distribution P of our target variables with respect to the input features, given some approximate distribution Q. 
    // This Can be achieved by minimizing the Dkl(P||Q) then it is called forward KL.
    // If we are minimizing Dkl(Q||P) then it is called backward KL.
    // Forward KL → applied in Supervised Learning
    // Backward KL → applied in Reinforcement learning
    template <FloatingTypes T = double>
    inline T KullbackLieblerDivergenceBackward(const std::vector<T>& output, const std::vector<T>& target) {
        T loss = 0.0;
        for (int i = 0, imax = output.size(); i < imax; ++i) {
            loss += output[i] * std::log(output[i] / target[i]);
        }
        return loss;
    };


    // Focal Loss
    // E({w[ij]}) = -(1/k_max) * Sum[k in output, k_max]((1 - output[i])^gamma * target[i] * log(output[i]))
    // (1 - output[i])^gamma is the weighting factor
    // Purpose: For Multi-Class Classification
    // Activation Function: 
    // Use the focal loss function in single-label classification tasks as an alternative to the more commonly used categorical crossentropy.
    template <FloatingTypes T = double>
    inline T FocalLoss(const std::vector<T>& output, const std::vector<T>& target, const double gamma) {
        const int output_size = output.size();
        T loss{ 0 };
        for (int i = 0; i < output_size; ++i) {
            loss += std::pow(1 - output[i], gamma) * target[i] * std::log(output[i]);
        }
        return loss;
    };
    // https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/focal-loss
    

    // Focal Loss
    // 1/2 * a^2                    ; for |a| <= delta
    // delta * (|a| - 1/2*delata)     otherwise
    // Purpose: For Multi-Class Classification
    // Activation Function: 
    // The Huber loss function describes the penalty incurred by an estimation procedure f. 
    template <FloatingTypes T = double>
    inline T HuberLoss(const std::vector<T>& output, const std::vector<T>& target, const double delta) {
        const int output_size = output.size();
        T loss{ 0 };
        T a{};
        for (int i = 0; i < output_size; ++i) {
            a = target[i] - output[i];
            if (std::abs(a) <= delta) { loss = std::pow(a, 2) / 2; }
            else { loss = delta * std::abs(a) - std::pow(delta, 2) / 2; }
        }
        return loss;
    };
    // https://en.wikipedia.org/wiki/Huber_loss

// !Multi-Class Single Label Classification Loss Functions

// Multi-Class Multi Label Classification Loss Functions
// !Multi-Class Multi Label Classification Loss Functions

}

#endif // !LOSS_FUNCTION_HPP_


// https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-squared-logarithmic-loss-(msle)

// Classification. Single label:Categorical crossentropy.Multi-label: Binary crossentropy, Squared hinge.
// Regression. Continuous values: Mean squared error, ​Mean absolute error,Mean squared logarithmic error. Discrete values: Poisson.

// https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/
// Regression Loss Functions. Output Layer Configuration: One node with a linear activation unit. Loss Function : Mean Squared Error(MSE).
    // Mean Squared Logarithmic Error Loss
    // Mean Absolute Error Loss
    
// Binary Classification Loss Functions. Output Layer Configuration: One node with a sigmoid activation unit. Loss Function : Cross - Entropy, also referred to as Logarithmic loss.
    // Binary Cross - Entropy
    // Hinge Loss
    // Squared Hinge Loss
    
// Multi-Class Classification Loss Functions. Output Layer Configuration: One node for each class using the softmax activation function. Loss Function : Cross - Entropy, also referred to as Logarithmic loss.
    // Multi - Class Cross - Entropy Loss
    // Sparse Multiclass Cross - Entropy Loss
    // Kullback Leibler Divergence Loss


// https://medium.com/@zeeshanmulla/cost-activation-loss-function-neural-network-deep-learning-what-are-these-91167825a4de
// Regression Loss Functions
    //Regression models deals with predicting a continuous value for example given floor area, number of rooms, size of rooms, predict the price of the room.The loss function used in the regression problem is called “Regression Loss Function”.
// Binary Classification Loss Functions
    // Binary classification is a prediction algorithm where the output can be either one of two items, indicated by 0 or 1. The output of binary classification algorithms is a prediction score (mostly). So the classification happens based on the threshold the value (default value is 0.5). If the prediction score > threshold then 1 else 0.
// Multi-Class Classification Loss Functions
    //Multi - Class classification are those predictive modeling problems where there are more target variables / class.It is just the extension of binary classification problem.
