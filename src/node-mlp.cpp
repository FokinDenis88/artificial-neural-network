#include "node-mlp.hpp"

#include <exception>
#include <stdexcept>
#include <random>
#include <cmath>

namespace mlp { //public

    void NodeMLP::ProcessNodeOutput() {
        // NotInline code block is faste than inline CalcNetInput
        net_input_ = 0;
        for (const Edge& edge : edges_) {
            net_input_ += (*edge.input_ptr) * edge.weight;
        }

        switch (activation_function_type_) {
            case ActivationFunctionType::Identity:
                output_ = fn_actv::IdentityFn<InputWeightOutputT>(net_input_);
                break;
            case ActivationFunctionType::Binary_step:
                output_ = fn_actv::BinaryStepFn<InputWeightOutputT>(net_input_);
                break;
            case ActivationFunctionType::Sigmoid_Logistic_soft_step:
                output_ = fn_actv::SigmoidLogisticFn<InputWeightOutputT>(net_input_);
                break;
            case ActivationFunctionType::Hyperbolic_tangent:
                output_ = fn_actv::HyperbolicTangentFn<InputWeightOutputT>(net_input_);
                break;
            case ActivationFunctionType::ReLU_Rectified_linear_unit:
                output_ = fn_actv::RectifiedLinearUnitFn<InputWeightOutputT>(net_input_);
                break;
            case ActivationFunctionType::GELU_Gaussian_Error_Linear_Unit:
                output_ = fn_actv::GaussianErrorLinearUnitFn<InputWeightOutputT>(net_input_);
                break;
            case ActivationFunctionType::Softplus:
                output_ = fn_actv::SoftplusFn<InputWeightOutputT>(net_input_);
                break;
            case ActivationFunctionType::ELU_Exponential_linear_unit:
                output_ = fn_actv::ExponentialLinearUnitFn<InputWeightOutputT>(net_input_, 1);
                break;
            case ActivationFunctionType::SELU_Scaled_exponential_linear_unit:
                output_ = fn_actv::ScaledExponentialLinearUnitFn<InputWeightOutputT>(net_input_);
                break;
            case ActivationFunctionType::Leaky_ReLU_Leaky_rectified_linear_unit:
                output_ = fn_actv::LeakyRectifiedLinearUnitFn<InputWeightOutputT>(net_input_);
                break;
            case ActivationFunctionType::PReLU_Parameteric_rectified_linear_unit:
                output_ = fn_actv::ParametricRectifiedLinearUnitFn<InputWeightOutputT>(net_input_, 1);
                break;
            case ActivationFunctionType::SiLU_Sigmoid_linear_unit:
                output_ = fn_actv::SigmoidLinearUnitFn<InputWeightOutputT>(net_input_);
                break;
            case ActivationFunctionType::Mish:
                output_ = fn_actv::MishFn<InputWeightOutputT>(net_input_);
                break;
            case ActivationFunctionType::Gaussian:
                output_ = fn_actv::GaussianFn<InputWeightOutputT>(net_input_);
                break;
                /*case ActivationFunctionType::Softmax:
                    output_ = fn_actv::softmax<InputWeightOutputT>(net_input_);
                    break;*/
            case ActivationFunctionType::Maxout:
                //output_ = fn_actv::MaxoutFn<InputWeightOutputT>(net_input_);
                break;
            case ActivationFunctionType::Linear:
                output_ = fn_actv::LinearFn<InputWeightOutputT>(1, net_input_, 0);
                break;
            case ActivationFunctionType::GaussianRBFFn:
                output_ = fn_actv::GaussianRBFFn<InputWeightOutputT>(net_input_, 1, 1);
                break;
            case ActivationFunctionType::HeavisideFn:
                output_ = fn_actv::HeavisideFn<InputWeightOutputT>(net_input_, 1, 1);
                break;
            case ActivationFunctionType::MultiquadraticsFn:
                output_ = fn_actv::MultiquadraticsFn<InputWeightOutputT>(net_input_, 1, 1);
                break;
        }
        //output_ = y_in;
    }

    void NodeMLP::RandomizeWeightsNode(double min, double max) {
        std::minstd_rand engine(std::random_device{}());
        std::uniform_real_distribution<> distribution(min, max);
        for (Edge& edge : edges_) {
            do { // weight must not be equal 0
                edge.weight = distribution(engine);
            } while (edge.weight == 0.0);
        }
    }

    void NodeMLP::RandomizeWeightsByNodesCount() {
        std::minstd_rand engine(std::random_device{}());
        double magnitude{1.0 / std::sqrt(edges_.size())};
        std::uniform_real_distribution<> distribution(-magnitude, magnitude);
        for (Edge& edge : edges_) {
            do { // weight must not be equal 0
                edge.weight = distribution(engine);
            } while (edge.weight == 0.0);
        }
    }

    void NodeMLP::ResetNode() {
        for (Edge& edge : edges_) {
            edge.weight = default_weight;
        }
    }

//private

//SupervisedLearnNodeMLP Neuron====================================================

// Unsupervised Learning

    void NodeMLP::HebbianLR(const double learning_rate_p) {
        ProcessNodeOutput(); // Any Activation Function
        for (Edge& edge : edges_) { // wi(t) = Nu * xi(t) * Y(t)
            edge.weight = learning_rate_p * (*edge.input_ptr) * output_;
        }
    }
    // https://en.wikipedia.org/wiki/Hebbian_theory
    // https://www.tutorialspoint.com/artificial_neural_network/artificial_neural_network_learning_adaptation.htm

    void NodeMLP::CompetitiveLR(const double learning_rate_p) {
        // TODO: finish learning rule
        // Condition to be a winner: yk = 1; vk > vj for all j       yk = 0; otherwise
        // Condition of sum total of weight: sumj(wkj) = 1; for all k
        for (Edge& edge : edges_) {
            // Change of weight for winner: delta_wkj = -Nu*(xj - wkj); if k wins        delta_wkj = 0 ; if k losses
            edge.weight += -learning_rate_p * ((*edge.input_ptr) - edge.weight);
        }
    }

// Supervised Learning

    void NodeMLP::CorrelationLR(const double learning_rate_p, const InputWeightOutputT target_output) {
        ProcessNodeOutput(); // Any Activation Function
        for (Edge& edge : edges_) { // delta_wij = Nu*xi*tj
            edge.weight = learning_rate_p * (*edge.input_ptr) * target_output;
        }
    }

    void NodeMLP::PerceptronLR(const double learning_rate_p, const InputWeightOutputT target_output) {
        // TODO: PerceptronLR will be used Only with BinaryStep activation function
        ProcessNodeOutput();
        if (target_output != output_) { // Change weights if necessary
            for (Edge& edge : edges_) { // w(new) = w(old) + Nu*(t-Y)*x
                edge.weight += learning_rate_p * (target_output - output_) * (*edge.input_ptr);
            }
        } // else: No change in weight
    }

    void NodeMLP::DeltaLR(const double learning_rate_p, const InputWeightOutputT target_output) {
        ProcessNodeOutput(); // Any Activation Function
        if (target_output != output_) { // Change weights if necessary
            for (Edge& edge : edges_) { // Same as PerceptronLR()
                // delta_wi = Nu * xi * ej
                // ej = (t - Y)
                // w(new) = w(old) + delta_w
                edge.weight += learning_rate_p * (target_output - output_) * (*edge.input_ptr);
            }
        } // else: No change in weight
    }

    void NodeMLP::OutstarLR(const double learning_rate_p, const InputWeightOutputT target_output) {
        ProcessNodeOutput(); // Any Activation Function
        if (target_output != output_) { // Change weights if necessary
            for (Edge& edge : edges_) {
                // delta_wj = Nu * (t - wj)
                edge.weight += learning_rate_p * (target_output - edge.weight);
            }
        } // else: No change in weight
    }

    void NodeMLP::BackPropagationAlgorithm() {
        ProcessNodeOutput(); // Sigmoid-Logistic Activation Function
    }

// Reinforcement

//!SupervisedLearnNodeMLP Neuron====================================================

} // !namespace mlp
