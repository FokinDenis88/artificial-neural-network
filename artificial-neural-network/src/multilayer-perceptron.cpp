#include "multilayer-perceptron.hpp"

#include <cmath>
#include <fstream>
//#include <exception>
//#include <stdexcept>
#include <execution>
#include <algorithm>
#include <iterator>
#include <mutex>
#include <type_traits>
#include <iostream>
#include <utility>
// for accumulate
#include <numeric>

// For testing code performance
#include <chrono>

#include <omp.h>

#include "BOM.h"
#include "FileService.h"
#include "WriteSerializedDataFile.h"
#include "ReadSerializedDataFile.h"
#include "ReadObjectInVecByte.h"

namespace mlp {
//public=================================
    
    MultiLayerPerceptron::MultiLayerPerceptron(const std::vector<size_t>& layers_dimension_p,
                                               NodeMLP::ActivationFunctionType hidden_layer_func_type_p,
                                               NodeMLP::ActivationFunctionType output_layer_func_type_p,
                                               LossFunctionType loss_func_type_p,
                                               LearningRateSchedule learning_rate_schedule_p,
                                               //const double permissible_prediction_error_p,
                                               const bool to_normalize_data_p,
                                               const bool has_bias_p)
        :   layers_dimension_{ layers_dimension_p },
            hidden_layer_actv_func_type_{ hidden_layer_func_type_p },
            output_layer_actv_func_type_{ output_layer_func_type_p },
            loss_func_type_{ loss_func_type_p },
            learning_rate_schedule_{ learning_rate_schedule_p },
            //permissible_prediction_error_{ permissible_prediction_error_p },
            to_normalize_data_{ to_normalize_data_p },
            has_bias_{ has_bias_p }
{
        if (layers_dimension_p.size() > 0) {
            ResizeLayersNInput();
            CreateAllEdges();
            SetAllActivationFuncs(hidden_layer_func_type_p, output_layer_func_type_p);
            SetAllNodesIndexes();

            // First step of learning algorithm is to randomize weights to small values
            // https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
            RandomizeAllWeightsByNodesCountMLP();
            //RandomizeWeightsMLP(-default_weight_randomization, default_weight_randomization);
            //RandomizeWeightsMLP(-weight_small_min, weight_small_min);

            Initialization();
        }
    }

    void MultiLayerPerceptron::ForwardPropagateMatrix() {
        if (weights_in_layer_.empty()) { // Initialize data for matrix calculations
            InitLearnMatrixMLP();
        }

        for (int i = 0, imax = struct_mlp_.nodes_layers_n_edges.size(); i < imax; ++i) {
            ProcessOutputOfLayerMtx(i);
        }
    }

    void MultiLayerPerceptron::ProcessOutputOfLayerMtx(size_t layer) {
        if (layer > 0) { // if not first layer
            output_of_layers_[layer] = weights_in_layer_[layer] * output_of_layers_[layer-1];
            // TODO: Добавить switch на все функции
            fn_actv::SigmoidLogisticFn(output_of_layers_[layer]);
            if (has_bias_ && layer < struct_mlp_.nodes_layers_n_edges.size() - 1) { // Not Last Layer
                auto last_row_index{ output_of_layers_[layer].rows() };
                output_of_layers_[layer].conservativeResize(last_row_index + 1);
                output_of_layers_[layer](last_row_index) = struct_mlp_.bias_output;
            }
        } else if (layer == 0) { // First layer
            size_t nodes_in_input_layer_count{ struct_mlp_.output_of_input_layer.size() };
            for (size_t i = 0; i < nodes_in_input_layer_count; ++i) {
                output_of_layers_[0](i) = struct_mlp_.output_of_input_layer[i];
            }
            // Bias output
            if (has_bias_) { output_of_layers_[0](nodes_in_input_layer_count) = struct_mlp_.bias_output; }
        } else { throw ErrorRuntimeMLP("Layer of neural network is below zero."); }
    }
  

// This Function is Less than 0.01% of performance of program
    void MultiLayerPerceptron::ForwardPropagateNode() {
        size_t last_layer_indx{ struct_mlp_.nodes_layers_n_edges.size() - 1 };

        for (int layer = 0, maxi = last_layer_indx; layer < maxi; ++layer) { // Process all layers except last layer
            std::for_each(std::execution::par_unseq, struct_mlp_.nodes_layers_n_edges[layer].begin(),
                          struct_mlp_.nodes_layers_n_edges[layer].end(), [](NodeMLP& node) { node.ProcessNodeOutput(); });
        }

        if (output_layer_actv_func_type_ != NodeMLP::ActivationFunctionType::Softmax) {
            // Process last layer
            std::for_each(std::execution::par_unseq, struct_mlp_.nodes_layers_n_edges[last_layer_indx].begin(),
                          struct_mlp_.nodes_layers_n_edges[last_layer_indx].end(), [](NodeMLP& node) { node.ProcessNodeOutput(); });
        } else { // Softmax activation function output layer

// Performance critical code block
            // Calc net input
            const int last_layer_indx = GetLastLayerIndex();
            const int nodes_count = layers_dimension_[last_layer_indx];
            int node{};
#pragma omp parallel for schedule(static)
            for (node = 0; node < nodes_count; ++node) { // CalcNetInput
                output_layer_net_input_vec_[node] = struct_mlp_.nodes_layers_n_edges[last_layer_indx][node].CalcNetInput();
            }
            NodeMLP::InputWeightOutputT exp_sum{ fn_actv::ExpSum(output_layer_net_input_vec_) };

#pragma omp parallel for schedule(static)
            for (node = 0; node < nodes_count; ++node) { // Calc outputs of output layer
                struct_mlp_.nodes_layers_n_edges[last_layer_indx][node].output_ = 
                                                                    fn_actv::SoftmaxFn(output_layer_net_input_vec_, node, exp_sum);
            }
// !Performance critical code block

        }
    }

    void MultiLayerPerceptron::SupervisedLearnMatrixMLP(const EpochIndexT iter_max_count, const TensorT& target) {
        // TODO: throw error
        // if (target.size() != layers_dimension_.back()) { throw ErrorRuntimeMLP("Error in dimension of target vector in SupervisedLearnNodeMLP."); }
        
        target_ = target;
        //InitLearnMatrixMLP();
        const EpochIndexT imax = current_epoch_ + iter_max_count;
        ForwardPropagateNode();
        ForwardPropagateMatrix(); // for first CheckPredictionError
        while (!CheckPredictionErrorMatrix() && current_epoch_ < imax) {
        //while (current_epoch_ < imax) {
            CalcLearningRate();
            ForwardPropagateMatrix();
            BackPropagationOfError();
            CalcWeightsMatrixForm();

            ++current_epoch_;
        }
        //SaveWeightsFrmMatrixToNodes();
    }

    void MultiLayerPerceptron::BackPropagationOfError() {
        size_t layers_count{ struct_mlp_.nodes_layers_n_edges.size() };
        size_t last_layer{ layers_count - 1 };
        error_in_layers_[last_layer] = fn_loss::MeanSquaredError(output_of_layers_[last_layer], target_vec_);
        int next_layer{};
        for (int layer = last_layer - 1; layer >= 0; --layer) { // Calculate Error from previous from last layer to previous
            next_layer = layer + 1;
                                  // E[prev] = W[between prev & currnet layer][Transponed M] * E[current layer]
            auto tw{ weights_in_layer_[next_layer].transpose() };
            error_in_layers_[layer] = tw * error_in_layers_[next_layer];
            //error_in_layers_[layer] = weights_in_layer_[next_layer].transpose() * error_in_layers_[next_layer];
        }
    }


    //Making min & max vectors for normalization of input values
    void MultiLayerPerceptron::FindMinMax(const std::vector<TensorT>& input_vec_p,
                    std::vector<NodeMLP::InputWeightOutputT>& min_input_p,
                    std::vector<NodeMLP::InputWeightOutputT>& max_input_p) {
        int columns_count = input_vec_p[0].size();
        size_t rows_count{ input_vec_p.size() };
        NodeMLP::InputWeightOutputT min{};
        NodeMLP::InputWeightOutputT max{};

#pragma omp parallel for schedule(static)
        for (int column = 0; column < columns_count; ++column) {
            min = input_vec_p[0][column];
            max = input_vec_p[0][column];
            for (size_t row = 1; row < rows_count; ++row) {
                if (input_vec_p[row][column] < min) {
                    min = input_vec_p[row][column];
                }
                if (input_vec_p[row][column] > max) {
                    max = input_vec_p[row][column];
                }
            }
            min_input_p[column] = min;
            max_input_p[column] = max;
        }
    }

    void MultiLayerPerceptron::NormalizeVector(std::vector<TensorT>& vec) {
        int rows_count = vec.size();
        if (rows_count > 1) { // Rows count in training vector > 1.
            size_t columns_count_in_input_vec{ vec[0].size() }; // number of neurons in input layer
            std::vector<NodeMLP::InputWeightOutputT> min_input(columns_count_in_input_vec);
            std::vector<NodeMLP::InputWeightOutputT> max_input(columns_count_in_input_vec);
            FindMinMax(vec, min_input, max_input);

            #pragma omp parallel for schedule(static)
            for (int row = 0; row < rows_count; ++row) {
                for (int column = 0, column_count = layers_dimension_[0]; column < column_count; ++column) {
                    if (min_input[column] != max_input[column]) { // if min = max NormalizeData will be nan & undefined / 0
                        struct_mlp_.output_of_input_layer[column] = 
                                                NormalizeValue(vec[row][column], min_input[column], max_input[column]);
                    }
                    else { struct_mlp_.output_of_input_layer = vec[row]; }
                }
            }
        }

    }

    void MultiLayerPerceptron::SupervisedLearnNodeMLP(std::vector<TensorT>& input, std::vector<TensorT>& target,
                                                      const EpochIndexT max_epoch_index_p, bool is_check_erorr_threshold) {
        if (input.size() == 0) { throw ErrorRuntimeMLP("Input vector is empty in SupervisedLearnNodeMLP."); }
        if (target.size() == 0) { throw ErrorRuntimeMLP("Target vector is empty in SupervisedLearnNodeMLP."); }
        if (input.size() != target.size()) {
            throw ErrorRuntimeMLP("Different dimensions of target vector & input vector in SupervisedLearnNodeMLP."); }
        if (input[0].size() != GetFirstLayerSize()) { throw ErrorRuntimeMLP("Error in dimension of input Tensor in SupervisedLearnNodeMLP."); }
        if (target[0].size() != GetLastLayerSize()) { throw ErrorRuntimeMLP("Error in dimension of target Tensor in SupervisedLearnNodeMLP."); }


        InitLearningProcessNode(input);   // Starts new process of network learning
        std::chrono::steady_clock::time_point start{};
        std::chrono::steady_clock::time_point end{};
        size_t tick_count{};
        max_epoch_index_ = max_epoch_index_p;
        size_t input_data_count{ input.size() };  // rows count
        
        NormalizeData(input, target);
        
        target_ = target[0];
        struct_mlp_.output_of_input_layer = input[0];
        ForwardPropagateNode(); // for first CheckPredictionError
        while (!to_check_prediction_error && current_epoch_ <= max_epoch_index_
               || to_check_prediction_error && !CheckPredictionError() && current_epoch_ <= max_epoch_index_) { // Iterate epochs
            if (to_show_learning_process_info) {
                WriteEpochInfoInConsole();
                if (current_epoch_ == 1) { start = std::chrono::steady_clock::now(); }
                else {
                    std::cout << "Time left: " << (max_epoch_index_ - current_epoch_ + 1) * tick_count * 1e-9 << " seconds\n";
                    if (!loss_fn_average_history_.empty()) {
                        std::cout << "Loss Function = " << loss_fn_average_history_.back() << '\n';
                    }
                }
            }

            CalcLearningRate();
            for (size_t row = 0; row < input_data_count; ++row) { // Iterate learning input data
            // TODO: 20% Can be optimized
                struct_mlp_.output_of_input_layer = input[row];
                target_ = target[row];      // !20%
                ForwardPropagateNode();     // 21%
                CalcCorrectionMatrix();     // 40%  big nodes count -> 30%
                CalcWeights();              // 27%  big nodes count -> 70%
            // 11%
                if (to_save_loss_fn_mean_history_ || !to_save_loss_fn_mean_history_ && current_epoch_ == max_epoch_index_) {
                    loss_fn_values_[row] = CalcLossFunction(GetOutputTensor(), target[row]);
                }
            // !11%
            }

            if (to_save_loss_fn_mean_history_ || !to_save_loss_fn_mean_history_ && current_epoch_ == max_epoch_index_) { // Save history
                // Mean, average of all loss values from input data set
                loss_fn_average_history_.emplace_back(std::accumulate(loss_fn_values_.begin(), 
                                                   loss_fn_values_.end(), static_cast<NetValuesType>(0)) / loss_fn_values_.size());
            }
            if (to_show_learning_process_info && current_epoch_ == 1) { // Show learning info
                end = std::chrono::steady_clock::now();
                tick_count = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            }
            ++current_epoch_;
        }
        CoutPredictionErrorPercent();
    }


    inline double PartPercent(const double part_of_value, const double value) {
        return (part_of_value / value * 100.0);
    }
    inline double PartPercent(const long double part_of_value, const long double value) {
        return (part_of_value / value * 100.0);
    }

    // if for_each_output == false then ErorrThreshold will calculate to summ of all errors
    // F.e. ErorrThreshold = 5%. So if all_outputs_magnitude == true, each output will has ErorrThreshold == 5%
    bool MultiLayerPerceptron::CheckPredictionError() {
        bool is_correct{ true };
        auto output_layer{ struct_mlp_.nodes_layers_n_edges.back() };
        if (is_error_for_each_output) { // each output will has f.e. permissible_prediction_error_ == 5%
            int i = 0, imax = output_layer.size();
            while (is_correct && i < imax) {
                prediction_error_percent_[i] = PartPercent(std::abs(target_[i] - output_layer[i].GetOutput()), target_[i]);
                if (prediction_error_percent_[i] > permissible_prediction_error_) { is_correct = false; }
                ++i;
            }
        } else { // permissible_prediction_error_ = 5% between all outputs
            NetValuesType target_sum{};
            for (int i = 0, imax = target_.size(); i < imax; ++i) {
                target_sum += target_[i];
            }
            prediction_error_percent_[0] = PartPercent( fn_loss::MeanAbsoluteError<NetValuesType>(GetOutputTensor(), target_), target_sum );
            if (prediction_error_percent_[0] > permissible_prediction_error_) {
                is_correct = false;
            }
        }
        return is_correct;
    }

    bool MultiLayerPerceptron::CheckPredictionErrorMatrix() {
        bool is_correct{ true };
//        auto output_of_mlp{ output_of_layers_.back() };
//        if (is_error_for_each_output) { // each output will has f.e. ErorrThreshold == 5%
//            int imax = output_of_mlp.size();
//#pragma omp parallel for schedule(static)
//            for (int i = 0; i < imax; ++i) {
//                error_percent_[i] = { PartPercent(std::abs(target_vec_(i) - output_of_mlp(i)), target_vec_(i)) };
//                if (error_percent_[i] > permissible_prediction_error_) { is_correct = false; }
//            }
//        } else { // 5% between all outputs
//            Eigen::VectorXd output{ output_of_mlp };
//            output = target_vec_ - output;
//            error_percent_[0] = PartPercent(std::abs(output.sum()), target_vec_.sum());
//            if (error_percent_[0] > permissible_prediction_error_) {
//                is_correct = false;
//            }
//        }
        return is_correct;
    }

    void MultiLayerPerceptron::RandomizeWeightsMLP(double min, double max) {
        for (OneLayerT& layer : struct_mlp_.nodes_layers_n_edges) {
            std::for_each(std::execution::par_unseq, layer.begin(), layer.end(), 
                          [&min, &max](NodeMLP& node) { node.RandomizeWeightsNode(min, max); });
        }
    }
    void MultiLayerPerceptron::RandomizeAllWeightsByNodesCountMLP() {
        for (OneLayerT& layer : struct_mlp_.nodes_layers_n_edges) {
            std::for_each(std::execution::par_unseq, layer.begin(), layer.end(),
                [](NodeMLP& node) { node.RandomizeWeightsByNodesCount(); });
        }
    }

    void MultiLayerPerceptron::LoadMLP(const std::string& file_name) {
        // TODO: Throw exception when error
        const std::string file_path{ kSavedMLP_Folder + file_name + kMLP_F_Extension };
        std::vector<unsigned char> readed_data{ file::ReadSerializedDataFile(file_path) };
        long long obj_first_byte_index{ 0 };

        // BOM
        //file::WriteBOM(write_file_stream, file::BOMEnum::No_BOM);
        
        // Topology
        file::ReadObjectInVecByte(&topology_, readed_data, obj_first_byte_index);

        // Hidden Layer Activation Function
        file::ReadObjectInVecByte(&hidden_layer_actv_func_type_, readed_data, obj_first_byte_index);
        
        // Output Layer Activation Function
        file::ReadObjectInVecByte(&output_layer_actv_func_type_, readed_data, obj_first_byte_index);

        // Loss Function Type
        file::ReadObjectInVecByte(&loss_func_type_, readed_data, obj_first_byte_index);

        // Bias
        file::ReadObjectInVecByte(&has_bias_, readed_data, obj_first_byte_index);

        // Normalization flag
        file::ReadObjectInVecByte(&to_normalize_data_, readed_data, obj_first_byte_index);

        // Learning Rate Schedule
        file::ReadObjectInVecByte(&learning_rate_schedule_, readed_data, obj_first_byte_index);

        // Initial Learning rate
        file::ReadObjectInVecByte(&learning_rate_initial_, readed_data, obj_first_byte_index);

        // Current Learning rate
        file::ReadObjectInVecByte(&learning_rate_, readed_data, obj_first_byte_index);

        // Iteration step
        file::ReadObjectInVecByte(&current_epoch_, readed_data, obj_first_byte_index);

        // Decay is needed for changing learning rate
        file::ReadObjectInVecByte(&decay_, readed_data, obj_first_byte_index);

        // Momentum
        file::ReadObjectInVecByte(&momentum_, readed_data, obj_first_byte_index);

        // Erorr threshold
        file::ReadObjectInVecByte(&permissible_prediction_error_, readed_data, obj_first_byte_index);

        // Mean Loss function value
        //file::ReadObjectInVecByte(&loss_fn_mean_, readed_data, obj_first_byte_index);

        // correction_matrix_ & delta_weights_ will be new for each epoch of learning

        // layers count
        size_t layers_count;
        file::ReadObjectInVecByte(&layers_count, readed_data, obj_first_byte_index);
        struct_mlp_.nodes_layers_n_edges.resize(layers_count);

        // count of neurons in layers
        std::vector<size_t> nodes_count_in_layers(layers_count);
        // TODO: Проверка на число элементов в массиве
        file::ReadObjectInVecByte(&nodes_count_in_layers[0], readed_data, obj_first_byte_index, sizeof(size_t) * layers_count);
        struct_mlp_.output_of_input_layer.resize(nodes_count_in_layers[0]);
        for (long i = 1; i < layers_count; ++i) { // enter empty nodes to layers
            struct_mlp_.nodes_layers_n_edges[i].resize(nodes_count_in_layers[i]);
        }
        CreateAllEdges();

        // Weights
        for (long long i = 0, imax = struct_mlp_.nodes_layers_n_edges.size(); i < imax; ++i) { // Layers
            for (long long j = 0, jmax = struct_mlp_.nodes_layers_n_edges[i].size(); j < jmax; ++j) { // Nodes
                // activation function
                NodeMLP::ActivationFunctionType activation_func;
                file::ReadObjectInVecByte(&activation_func, readed_data, obj_first_byte_index);
                struct_mlp_.nodes_layers_n_edges[i][j].SetActivationFunctionType(activation_func);

                // read node weights count
                size_t node_weights_count;
                file::ReadObjectInVecByte(&node_weights_count, readed_data, obj_first_byte_index);
                struct_mlp_.nodes_layers_n_edges[i][j].ResizeEdgesVec(node_weights_count);

                // load weights
                std::vector<NodeMLP::InputWeightOutputT> node_weights(node_weights_count);
                file::ReadObjectInVecByte(&node_weights[0], readed_data, obj_first_byte_index, 
                                            sizeof(NodeMLP::InputWeightOutputT) * node_weights_count);
                for (long k = 0; k < node_weights_count; ++k) {
                    struct_mlp_.nodes_layers_n_edges[i][j].SetWeight(k, node_weights[k]);
                }
            }
        }

        SetAllNodesIndexes();
    }

    void MultiLayerPerceptron::SaveMLP(const std::string& file_name) {
        // TODO: Throw exception when error
        if (struct_mlp_.nodes_layers_n_edges.size() > 0) { // TODO: Проверка корректности создания нейронной сети
            const std::string file_path{ kSavedMLP_Folder + file_name + kMLP_F_Extension};
            std::basic_ofstream<unsigned char> write_file_stream{
                                    file::OpenFile<std::basic_ofstream<unsigned char>>(file_path, file::OpenModeWriteBinaryRewrite) };
            // BOM
            //file::WriteBOM(write_file_stream, file::BOMEnum::No_BOM);
            
            // Topology
            file::WriteSerializedDataFile(write_file_stream, &topology_);

            // Hidden Layer Activation Function
            file::WriteSerializedDataFile(write_file_stream, &hidden_layer_actv_func_type_);

            // Output Layer Activation Function
            file::WriteSerializedDataFile(write_file_stream, &output_layer_actv_func_type_);

            // Loss Function Type
            file::WriteSerializedDataFile(write_file_stream, &loss_func_type_);

            // Bias
            file::WriteSerializedDataFile(write_file_stream, &has_bias_);

            // Normalization flag
            file::WriteSerializedDataFile(write_file_stream, &to_normalize_data_);

            // Learning Rate Schedule
            file::WriteSerializedDataFile(write_file_stream, &learning_rate_schedule_);

            // Initial Learning rate
            file::WriteSerializedDataFile(write_file_stream, &learning_rate_initial_);

            // Current Learning rate
            file::WriteSerializedDataFile(write_file_stream, &learning_rate_);

            // Iteration step
            file::WriteSerializedDataFile(write_file_stream, &current_epoch_);

            // Decay is needed for changing learning rate
            file::WriteSerializedDataFile(write_file_stream, &decay_);

            // Momentum
            file::WriteSerializedDataFile(write_file_stream, &momentum_);

            // Erorr threshold
            file::WriteSerializedDataFile(write_file_stream, &permissible_prediction_error_);

            // Mean Loss function value
            //file::WriteSerializedDataFile<unsigned char>(write_file_stream, &loss_fn_mean_);

            // correction_matrix_ & delta_weights_ will be new for each epoch of learning

            // layers count
            const size_t layers_count{ struct_mlp_.nodes_layers_n_edges.size() };
            file::WriteSerializedDataFile(write_file_stream, &layers_count);

            // count of neurons in layers
            std::vector<size_t> nodes_count_in_layers;
            nodes_count_in_layers.reserve(layers_count);
            nodes_count_in_layers.emplace_back(struct_mlp_.output_of_input_layer.size());
            for (long i = 1; i < layers_count; ++i) {
                nodes_count_in_layers.emplace_back(struct_mlp_.nodes_layers_n_edges[i].size());
            }
            file::WriteSerializedDataFile(write_file_stream, &nodes_count_in_layers[0], file::SizeOfArray(nodes_count_in_layers, layers_count));

            // Weights
            for (long long i = 0, imax = struct_mlp_.nodes_layers_n_edges.size(); i < imax; ++i) { // Layers
                for (long long j = 0, jmax = struct_mlp_.nodes_layers_n_edges[i].size(); j < jmax; ++j) { // Nodes
                    // activation function
                    const NodeMLP::ActivationFunctionType activation_func{ struct_mlp_.nodes_layers_n_edges[i][j].GetActivationFunctionType() };
                    file::WriteSerializedDataFile(write_file_stream, &activation_func);

                    // write weights count in current NodeMLP
                    std::vector<NodeMLP::InputWeightOutputT> node_weights{ struct_mlp_.nodes_layers_n_edges[i][j].GetWeights() };
                    size_t node_weights_count{ node_weights.size() };
                    file::WriteSerializedDataFile(write_file_stream, &node_weights_count);

                    // write weights of NodeMLP
                    file::WriteSerializedDataFile(write_file_stream, &node_weights[0], file::SizeOfArray(node_weights, node_weights_count));
                }
            }
            file::CloseFile(write_file_stream);
        }
    }

    void MultiLayerPerceptron::ResetMLP() {
        for (OneLayerT& layer : struct_mlp_.nodes_layers_n_edges) {
            std::for_each(std::execution::par_unseq, layer.begin(), layer.end(),
                          [](NodeMLP& node) { node.ResetNode(); });
        }
    }

    void MultiLayerPerceptron::ResizeLayersNInput() {
        size_t layers_count{ layers_dimension_.size() };
        struct_mlp_.nodes_layers_n_edges.resize(layers_count);
        // first layer of nodes_layers_n_edges is empty, because its functions are executed by output_of_input_layer
#pragma omp parallel for schedule(static)
        for (int i = 1; i < layers_count; ++i) {
            struct_mlp_.nodes_layers_n_edges[i].resize(layers_dimension_[i]);
        }

        struct_mlp_.output_of_input_layer.resize(layers_dimension_[0]);
    }

//private=================================

    void MultiLayerPerceptron::CreateAllEdges() {
        ConnectSecondLayerToInput();
        for (size_t layer = 2, imax = struct_mlp_.nodes_layers_n_edges.size(); layer < imax; ++layer) { // Layer
            int nodes_count = struct_mlp_.nodes_layers_n_edges[layer].size();
#pragma omp parallel for schedule(static)
            for (int node = 0; node < nodes_count; ++node) { // Nodes in Layer
                size_t prev_layer{ layer - 1 };
                size_t prev_nodes_count{ struct_mlp_.nodes_layers_n_edges[prev_layer].size() };
#pragma omp parallel for schedule(static)
                for (int prev_node = 0; prev_node < prev_nodes_count; ++prev_node) { // add links to all output of prev layer
                    struct_mlp_.nodes_layers_n_edges[layer][node].AddEdgeToNode(struct_mlp_.nodes_layers_n_edges[prev_layer][prev_node]);
                }

                // Add link to Bias for all layers except first layer. There is no links from first layer to bias
                if (has_bias_) { struct_mlp_.nodes_layers_n_edges[layer][node].AddEdgeToBias(&struct_mlp_.bias_output); }
            }
            //for (NodeMLP& node : struct_mlp_.nodes_layers_n_edges[layer]) { // Nodes in Layer
            //    size_t prev_layer{ layer - 1 };
            //    size_t prev_nodes_count{ struct_mlp_.nodes_layers_n_edges[prev_layer].size() };
            //    for (int prev_node = 0; prev_node < prev_nodes_count; ++prev_node) { // add links to all output of prev layer
            //        node.AddEdgeToNode(struct_mlp_.nodes_layers_n_edges[prev_layer][prev_node]);
            //    }

            //    // Add link to Bias for all layers except first layer. There is no links from first layer to bias
            //    if (has_bias_) { node.AddEdgeToBias(&struct_mlp_.bias_output); }
            //}
        }
    }

    void MultiLayerPerceptron::ConnectSecondLayerToInput() {
        if (struct_mlp_.nodes_layers_n_edges.size() > 1) {
            int nodes_count = struct_mlp_.nodes_layers_n_edges[1].size();
#pragma omp parallel for schedule(static)
            for (int node = 0; node < nodes_count;++node) { // nodes in second layer
            //for (NodeMLP& node : struct_mlp_.nodes_layers_n_edges[1]) { // nodes in second layer
                //node.DeleteAllEdges(); // clear previous edges 
                for (NodeMLP::InputWeightOutputT& input : struct_mlp_.output_of_input_layer) { // input
                    struct_mlp_.nodes_layers_n_edges[1][node].AddEdgeToInput(&input);
                }
                if (has_bias_) {
                    struct_mlp_.nodes_layers_n_edges[1][node].AddEdgeToBias(&struct_mlp_.bias_output);
                }
            }
        }
    }

    void MultiLayerPerceptron::SetAllActivationFuncs(NodeMLP::ActivationFunctionType hidden_layer_func_type_p,
                                                     NodeMLP::ActivationFunctionType output_layer_func_type_p) {
        size_t output_layer_indx{ struct_mlp_.nodes_layers_n_edges.size() - 1 };
        // Hidden layer
        for (size_t layer = 0; layer < output_layer_indx; ++layer) { // For all layers except last output layer
            for (NodeMLP& node : struct_mlp_.nodes_layers_n_edges[layer]) {
                node.SetActivationFunctionType(hidden_layer_func_type_p);
            }
        }
        // Output layer
        for (NodeMLP& node : struct_mlp_.nodes_layers_n_edges[output_layer_indx]) {
            node.SetActivationFunctionType(output_layer_func_type_p);
        }
    }

    void MultiLayerPerceptron::SetAllNodesIndexes() {
        for (OneLayerT& layer : struct_mlp_.nodes_layers_n_edges) {
            for (size_t i = 0, imax = layer.size(); i < imax; ++i) {
                layer[i].SetIndex(i);
            }
        }
    }

    void MultiLayerPerceptron::CalcPredictionErrorPercent() {
        auto output_layer{ struct_mlp_.nodes_layers_n_edges.back() };
        if (is_error_for_each_output) { // each output will has f.e. permissible_prediction_error_ == 5%
            int imax = output_layer.size();
#pragma omp parallel for schedule(static)
            for (int i = 0; i < imax; ++i) {
                prediction_error_percent_[i] = PartPercent(std::abs(target_[i] - output_layer[i].GetOutput()), target_[i]);
            }
        } else { // permissible_prediction_error_ = 5% between all outputs
            /*NetValuesType target_sum{};
            for (int i = 0, imax = target_.size(); i < imax; ++i) {
                target_sum += target_[i];
            }
            prediction_error_percent_[0] = PartPercent(fn_loss::MeanAbsoluteError<NetValuesType>(GetOutputTensor(), target_), target_sum);*/
        }
    }

    void mlp::MultiLayerPerceptron::CoutPredictionErrorPercent() {
        // TODO: Calc MeanAbsoluteError for all inputs of last epoch
        std::cout << "\nMean absolute error = " << fn_loss::MeanAbsoluteError(GetOutputTensor(), target_) << '\n';
        CalcPredictionErrorPercent();
        std::cout << "\nMean absolute error(% percent):\n";
        for (const double& elem : prediction_error_percent_) {
            std::cout << elem << '\n';
        }
        std::cout << "\n";
    }


    /*void MultiLayerPerceptron::InitInputVector() {
        size_t size{ struct_mlp_.output_of_input_layer.size() };
        input_vec_.resize(size);
#pragma omp parallel for schedule(static)
        for (int column = 0; column < size; ++column) {
            input_vec_(column) = struct_mlp_.output_of_input_layer[column];
        }
    }*/

    void MultiLayerPerceptron::Initialization() {
        InitLearnNodeMLP();
        InitLearnMatrixMLP();
    }

    void MultiLayerPerceptron::InitTargetVector() {
        size_t size{ target_.size() };
        target_vec_.resize(size);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < size; ++i) {
            target_vec_(i) = target_[i];
        }
    }

    void MultiLayerPerceptron::InitWeightsMatrix() {
        int layers_count = struct_mlp_.nodes_layers_n_edges.size();
        weights_in_layer_.resize(layers_count);
        // weights between input layer and input of network dont change. So weights_in_layer_[0] is empty
        size_t bias_node = has_bias_ ? 1 : 0;

#pragma omp parallel for schedule(static)
        for (int layer = 1; layer < layers_count; ++layer) { // iterate layers in nodes_layers_
            // There is no edge between bias in current layer & previous layer
            // rows count = count of nodes in current layer; columns count = count of nodes in previous layer
            size_t rows_count{ struct_mlp_.nodes_layers_n_edges[layer].size() };
            // columns count = count of nodes in previous layer
            size_t columns_count{ layer != 1 ? struct_mlp_.nodes_layers_n_edges[layer - 1].size() + bias_node
                                             : struct_mlp_.output_of_input_layer.size() + bias_node };
            weights_in_layer_[layer].resize(rows_count, columns_count);

#pragma omp parallel for schedule(static)
            for (int i_m = 0; i_m < rows_count; ++i_m) { // current layer
#pragma omp parallel for schedule(static)
                for (int j_m = 0; j_m < columns_count; ++j_m) { // i_m is nodes in current layer; j_m is nodes in previous layer
                    weights_in_layer_[layer](i_m, j_m) = GetWeightIJ(layer, j_m, i_m);
                }
            }
        }
    }

    void MultiLayerPerceptron::InitOutputMatrix() {
        size_t last_layer{ struct_mlp_.nodes_layers_n_edges.size() - 1 }; // Input layer is included
        output_of_layers_.resize(last_layer + 1);
        size_t bias_node = has_bias_ ? 1 : 0;

        // Input Layer
        output_of_layers_[0].resize(struct_mlp_.output_of_input_layer.size() + bias_node);
        // Hidden Layers
        for (int i = 1; i < last_layer; ++i) { // resize to count of nodes in layer
            output_of_layers_[i].resize(struct_mlp_.nodes_layers_n_edges[i].size() + bias_node);
        }
        // Output Layer
        output_of_layers_[last_layer].resize(struct_mlp_.nodes_layers_n_edges[last_layer].size());
    }

    void MultiLayerPerceptron::InitErrorMatrix() {
        size_t last_layer{ struct_mlp_.nodes_layers_n_edges.size() - 1 };
        error_in_layers_.resize(last_layer + 1);
        size_t bias_node = has_bias_ ? 1 : 0;

        // Input Layer
        error_in_layers_[0].resize(struct_mlp_.output_of_input_layer.size() + bias_node);
        // Hidden Layers
        for (int layer = 1; layer < last_layer; ++layer) { // Start from second layer
            error_in_layers_[layer].resize(struct_mlp_.nodes_layers_n_edges[layer].size() + bias_node);
        }
        // Output Layer
        error_in_layers_[last_layer].resize(struct_mlp_.nodes_layers_n_edges[last_layer].size());
    }

    void MultiLayerPerceptron::InitLearnMatrixMLP() {
        //InitInputVector();
        InitTargetVector();
        InitWeightsMatrix();
        InitOutputMatrix();
        InitErrorMatrix();
    }


//SupervisedLearnNodeMLP Rate Schedule====================================================
// https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
//https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/

    void MultiLayerPerceptron::CalcLearningRate() {
        switch (learning_rate_schedule_) {
        case LearningRateSchedule::Time_based:      TimeBasedLRateSchedule();       break;
        case LearningRateSchedule::Step_based:      StepBasedLRateSchedule();      break;
        case LearningRateSchedule::Exponential:     ExponentialLRateSchedule();     break;
        case LearningRateSchedule::Constant:        ConstantLRate();                break;
        }
    }

    void MultiLayerPerceptron::ConstantLRate() {
    }
    
    void MultiLayerPerceptron::TimeBasedLRateSchedule() {
                               // Nu[n+1] = Nu[n] / (1 + d*n)
        learning_rate_ = learning_rate_/(1 + decay_ * current_epoch_);
    }

    void MultiLayerPerceptron::StepBasedLRateSchedule() { 
                                        // Nu[n] = Nu[0]*d^(floor((1 + n)/r))
        learning_rate_ = learning_rate_initial_ * std::pow(decay_, std::floor((1 + current_epoch_) / epochs_drop_rate_));
    }

    void MultiLayerPerceptron::ExponentialLRateSchedule() {
                                    // Nu[n] = Nu[0]*exp^(-d*n)
        learning_rate_ = learning_rate_initial_ * std::exp(-decay_ * current_epoch_);
    }

    void MultiLayerPerceptron::InitCorrectionMtx() {
        size_t layers_count{ struct_mlp_.nodes_layers_n_edges.size() };
        if (correction_matrix_.size() != layers_count) {
            correction_matrix_.resize(layers_count);
            if (has_bias_) { // There is additional delta correction value on bias node
                correction_matrix_[0].resize(struct_mlp_.output_of_input_layer.size() + 1);
                size_t last_layer_index{ layers_count - 1 };
                for (int layer = 1, imax = last_layer_index; layer < imax; ++layer) { // iterate layers
                    correction_matrix_[layer].resize(struct_mlp_.nodes_layers_n_edges[layer].size() + 1);
                }
                // There is no bias in last layer
                correction_matrix_[last_layer_index].resize(struct_mlp_.nodes_layers_n_edges[last_layer_index].size());
            } else { // For non bias network
                correction_matrix_[0].resize(struct_mlp_.output_of_input_layer.size());
                for (int layer = 1, imax = layers_count; layer < imax; ++layer) { // iterate layers
                    correction_matrix_[layer].resize(struct_mlp_.nodes_layers_n_edges[layer].size());
                }
            }
        }
    }

    void MultiLayerPerceptron::InitDeltaWeightsMtx() {
        size_t layers_count{ struct_mlp_.nodes_layers_n_edges.size() };
        if (delta_weights_.size() != layers_count) {
            delta_weights_.resize(layers_count);
            // There is no delta weights between first layer of mlp and input vector. So delta_weights_[0] is empty.
            for (int layer = 1, imax = layers_count; layer < imax; ++layer) { // iterate layers
                int nodes_count = struct_mlp_.nodes_layers_n_edges[layer].size();
                delta_weights_[layer].resize(nodes_count);

#pragma omp parallel for schedule(static)
                for (int node = 0; node < nodes_count; ++node) {
                    delta_weights_[layer][node].resize(struct_mlp_.nodes_layers_n_edges[layer][node].GetEdgesCount());
                }
            }
        }
    }

    void MultiLayerPerceptron::InitLearnNodeMLP() {
        InitCorrectionMtx();
        InitDeltaWeightsMtx();
        size_t last_layer_size{ GetLastLayerSize() };
        output_layer_net_input_vec_.resize(last_layer_size);
        ResizePredictionErrorPercent();
    }

// This function is 70% of performance of program
    void MultiLayerPerceptron::CalcCorrectionMatrix() {
        // There is additional bias node in network
        size_t layers_count{ struct_mlp_.nodes_layers_n_edges.size() };
        size_t last_layer_indx{ layers_count - 1 };

        // Calculate correction for all nodes of last layer
        // There is no bias in last layer
        {
            int last_layer_nodes_count = struct_mlp_.nodes_layers_n_edges[last_layer_indx].size();
            if (output_layer_actv_func_type_ == NodeMLP::ActivationFunctionType::Softmax &&
                    loss_func_type_ == LossFunctionType::Categorical_Crossentropy) { // Only Softmax last layer act func & cross entropy loss
                // https://www.mldawn.com/back-propagation-with-cross-entropy-and-softmax/
                // https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
#pragma omp parallel for schedule(static)
                for (int ll_node_indx = 0; ll_node_indx < last_layer_nodes_count; ++ll_node_indx) { // ll_node_indx = last layer node index
                    NodeMLP::InputWeightOutputT output{ struct_mlp_.nodes_layers_n_edges[last_layer_indx][ll_node_indx].GetOutput() };
                                                        // dE / dZ[k] = o[k] - t[k]
                    correction_matrix_[last_layer_indx][ll_node_indx] = output - target_[ll_node_indx];
                }
            } else { // For every other case
#pragma omp parallel for schedule(static)
                for (int ll_node_indx = 0; ll_node_indx < last_layer_nodes_count; ++ll_node_indx) { // ll_node_indx = last layer node index
                    NodeMLP::InputWeightOutputT output{ struct_mlp_.nodes_layers_n_edges[last_layer_indx][ll_node_indx].GetOutput() };
                    // Derivative Activation Function Output layer * Derivative Loss Function
                                                   // (df(s) / dS) * (dE / do[j])   ; S = Sum to all weight*input
                                     // delt[k] = -o[k]*(1 - o[k]) * (t[k] - o[k])
                    correction_matrix_[last_layer_indx][ll_node_indx] =
                        CalcActivationFunctionDeriv(output_layer_actv_func_type_, output)
                        * CalcLossFunctionDerivRespectOutputJ(output, target_[ll_node_indx]);
                                                 // Derivative Activation Function Output layer * Derivative Loss Function
                    //correction_matrix_[last_layer_indx][ll_node_indx] = output * (1 - output) * (output - target_[ll_node_indx]);
                }
            }
        }

        // for all other layers except the last layer
        for (int layer = layers_count - 2; layer >= 0; --layer) {
            const int next_layer{ layer + 1 };
            size_t j_nodes_count{};
            if (layer != 0) { // Not First layer
                j_nodes_count = struct_mlp_.nodes_layers_n_edges[layer].size();
            } else { // First layer
                j_nodes_count = struct_mlp_.output_of_input_layer.size();
            }
            if (has_bias_) { // If there is additional bias node in network
                ++j_nodes_count;
            }

// This loop is 70% of performance of program
#pragma omp parallel for schedule(static)
            for (long j_node = 0; j_node < j_nodes_count; ++j_node) {
            // 50-60%
                NodeMLP::InputWeightOutputT output{};
                int bias_index = j_nodes_count - 1;
                if (has_bias_ && j_node == bias_index) { // Need to get output of bias node. Bias is in last edge.
                    output = struct_mlp_.bias_output;
                } else {
                    if (layer != 0) { // Not First layer
                        output = struct_mlp_.nodes_layers_n_edges[layer][j_node].GetOutput();
                    } else { // First layer
                        output = struct_mlp_.output_of_input_layer[j_node];
                    }
                }
                
                NodeMLP::InputWeightOutputT sum{};
                // There is no link between bias of next layer & previous layer
                // So there is no bias nodes in children of current node
                std::vector<NodeMLP::InputWeightOutputT> sum_values(struct_mlp_.nodes_layers_n_edges[next_layer].size());
            // !50-60%

                // Calculate Sum of delt & w[j,k] of next layer : Backpropagate calculated values of correction of last layer to first layer
                { // 20%
                    int next_layer_nodes_count = struct_mlp_.nodes_layers_n_edges[next_layer].size();
#pragma omp parallel for schedule(static)
                    for (int k = 0; k < next_layer_nodes_count; ++k) {
                    //for (const NodeMLP& node_in_next_layer : struct_mlp_.nodes_layers_n_edges[next_layer]) {
                        //k = node_in_next_layer.index_;
                        // delt[j] = o[j] * (1 - o[j]) * Sum[k in childer(j)](delt[k] * w[j, k])
                                                // Sum[k in childer(j)](delt[k] * w[j, k])
                        sum_values[k] = correction_matrix_[next_layer][k] * 
                                        struct_mlp_.nodes_layers_n_edges[next_layer][k].edges_[j_node].weight;
                        // Original Algorithm
                        //size_t k{ node_in_next_layer.GetIndex() };
                        //// delt[j] = o[j] * (1 - o[j]) * Sum[k in childer(j)](delt[k] * w[j, k])
                        //                        // Sum[k in childer(j)](delt[k] * w[j, k])
                        //sum_values[k] = correction_matrix_[next_layer][k] * GetWeightIJ(next_layer, j_node, k);
                    }
                } // !20%

            // 20%
                for (NodeMLP::InputWeightOutputT& sum_res : sum_values) {
                    sum += sum_res;
                } // Worser performance variants:
                // 1) #pragma omp parallel for schedule(static) reduction(+ : sum)
                // 2) sum = std::accumulate(sum_values.begin(), sum_values.end(), static_cast<NetValuesType>(0)); // lower performance
            // !20%
            // 20%
                if (has_bias_ && j_node == bias_index) { // If the node is bias. Derivative of bias activation func equal 1.
                                           // delt[j] = f(Identity)' * Sum[k in childer(j)](delt[k]w[j, k]);
                                           // delt[j] = 1 * Sum[k in childer(j)](delt[k]w[j, k]);
                    correction_matrix_[layer][j_node] = sum;
                } else {                   // Derivative of Activation Function Hidden layer * Sum
                                                        // (df(s) / dS) * Sum    ; 
                                           // delt[j] = o[j] * (1 - o[j]) * Sum[k in childer(j)](delt[k]w[j, k])
                    correction_matrix_[layer][j_node] = CalcActivationFunctionDeriv(hidden_layer_actv_func_type_, output) * sum;
                    //correction_matrix_[layer][j_node] = output * (1 - output) * sum;
                }
            // !20%
            }
        }
    }

// This Function is 30%-80% of Performance of program
// Critical performance code block when number of nodes is very high
    void MultiLayerPerceptron::CalcWeights() {
        // first layer has no edge with changes in weight. All delda_w in first layer = 0
        int layers_count = struct_mlp_.nodes_layers_n_edges.size();
        // Weights in First Input layer are not changing, they are const
//#pragma omp parallel for schedule(static) -> don't use. Or it will be low performance
        for (int layer = 1; layer < layers_count; ++layer) { // layer
            int prev_layer{ layer - 1 };
            // There is no link between bias of next layer & previous layer
            // There is no bias in last layer
            size_t count_current_layer_nodes_with_edges_to_prev_layer{ struct_mlp_.nodes_layers_n_edges[layer].size() };

            // 99% of performance of CalcWeights
#pragma omp parallel for schedule(static)
            for (int j_node = 0; j_node < count_current_layer_nodes_with_edges_to_prev_layer; ++j_node) { 
            // 28 - 45%
                //j = node in current layer with link to prev
                int nodes_prev_layer_count = has_bias_ ? layers_dimension_[prev_layer] + 1 : layers_dimension_[prev_layer];
                /*if (prev_layer != 0) { nodes_prev_layer_count = struct_mlp_.nodes_layers_n_edges[prev_layer].size(); }
                else { nodes_prev_layer_count = struct_mlp_.output_of_input_layer.size(); }
                if (prev_layer != 0) { nodes_prev_layer_count = layers_dimension_[prev_layer]; }
                else { nodes_prev_layer_count = layers_dimension_[0]; }
                if (has_bias_) { ++nodes_prev_layer_count; }*/
            // !28 - 45%

                // 70%
#pragma omp parallel for schedule(static)
                for (int i_node = 0; i_node < nodes_prev_layer_count; ++i_node) { // node = column in previous layer
                // 20%
                    NodeMLP::InputWeightOutputT& dw{ delta_weights_[layer][j_node][i_node] };
                    NodeMLP::InputWeightOutputT output_i{};
                    if (prev_layer != 0) { // Not Second layer
                        if (has_bias_ && i_node == nodes_prev_layer_count - 1) { // Is Bias
                            output_i = struct_mlp_.bias_output;
                        } else { // Is not Bias
                            output_i = struct_mlp_.nodes_layers_n_edges[prev_layer][i_node].GetOutput();
                        }
                    } else { // Is Second layer
                        if (has_bias_ && i_node == nodes_prev_layer_count - 1) { // Is Bias
                            output_i = struct_mlp_.bias_output;
                        } else { // Is not Bias
                            output_i = struct_mlp_.output_of_input_layer[i_node];
                        }
                    }
                // !20%

    // delta_w[column,j](n) =         momentum * delta_w[column,j](n-1) + (1-momentum)*Nu*delt[j]*o[column]
                    dw = momentum_ * dw + (1 - momentum_) * learning_rate_ * correction_matrix_[layer][j_node] * output_i;  // 20%
                
                                                                    // w[column,j](n) = w[column,j](n-1) - delta_w[column,j](n)
                    NodeMLP::InputWeightOutputT new_weight{ GetWeightIJ(layer, i_node, j_node) - dw };  // 30%
                    SetWeightIJ(layer, i_node, j_node, new_weight); // 30%
                } 
                // !70%
            } // !99%
        }
    }

    void MultiLayerPerceptron::CalcWeightsMatrixForm() {
        for (int k = 1, layers_count = weights_in_layer_.size(); k < layers_count; ++k) {
            auto temp{ output_of_layers_[k] };
            temp.array() -= 1.0;
                     // new W[jk] = old_W[jk] - Nu * E[k] * O[k]*(1 - O[k]) * O[j][transpont]
            weights_in_layer_[k] -= learning_rate_ * error_in_layers_[k] * output_of_layers_[k] * (-temp) * output_of_layers_[k-1].transpose();
        }
    }

    void MultiLayerPerceptron::SaveWeightsFrmMatrixToNodes() {
        for (int layer = 0, imax = weights_in_layer_.size(); layer < imax; ++layer) {
            for (int row = 0, nodes_count = weights_in_layer_[layer].rows(); row < nodes_count; ++row) {
                for (int col = 0, col_max = weights_in_layer_[layer].cols(); col < col_max; ++col) {
                    struct_mlp_.nodes_layers_n_edges[layer][row].SetWeight(col, weights_in_layer_[layer](row, col));
                }
            }
        }
    }

//!SupervisedLearnNodeMLP Rate Schedule====================================================


// Helper Functions in class

    void MultiLayerPerceptron::ScaleInput(TensorT& input_ptr, NodeMLP::InputWeightOutputT a, NodeMLP::InputWeightOutputT b) {
        std::for_each(std::execution::par_unseq, input_ptr.begin(), input_ptr.end(),
            [a, b](NodeMLP::InputWeightOutputT& value) { value *= a / b; });
    }
    

    // Processing and Displaying one input vector. Node Form of processing.
    void MultiLayerPerceptron::ProcessNDisplayOneInputMtrx(const std::vector<NodeMLP::InputWeightOutputT>& input_vec) {
        ForwardPropagateMatrix(input_vec);
        auto output{ GetOutputOfLayerVec(GetLayersCount() - 1) };
        std::cout << "Result = ";
        for (int i = 0, imax = output.size(); i < imax; ++i) {
            std::cout << output(i) << ' ';
        }
        std::cout << '\n';
    }
    // Processing and Displaying many input vectors. Node Form of processing.
    void MultiLayerPerceptron::ProcessNDisplayAllInputMtrx(const std::vector<std::vector<NodeMLP::InputWeightOutputT>>& all_input_vec) {
        for (const std::vector<NodeMLP::InputWeightOutputT>& input_vec : all_input_vec) { ProcessNDisplayOneInputMtrx(input_vec); }
        std::cout << '\n';
    }

    // Processing and Displaying one input vector. Node Form of processing.
    void MultiLayerPerceptron::ProcessNDisplayOneInput(const std::vector<NodeMLP::InputWeightOutputT>& input_vec) {
        ForwardPropagateNode(input_vec);
        auto output{ GetOutputTensor() };
        std::cout << "Result = ";
        for (int i = 0, imax = output.size(); i < imax; ++i) {
            std::cout << output[i] << ' ';
        }
        std::cout << '\n';
    }
    // Processing and Displaying many input vectors. Node Form of processing.
    void MultiLayerPerceptron::ProcessNDisplayAllInput(const std::vector<std::vector<NodeMLP::InputWeightOutputT>>& all_input_vec) {
        for (const std::vector<NodeMLP::InputWeightOutputT>& input_vec : all_input_vec) { ProcessNDisplayOneInput(input_vec); }
        std::cout << '\n';
    }

// !Helper Functions in class


// Helper Functions

    

} // !mlp