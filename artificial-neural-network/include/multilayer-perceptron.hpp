#ifndef MULTILAYER_PERCEPTRON_HPP_
#define MULTILAYER_PERCEPTRON_HPP_

#include <vector>
#include <string>
// For WriteEpochInfoInConsole
#include <stdlib.h>
#include <iostream>

#include "Eigen/Core"

#include "node-mlp.hpp"
#include "errors-mlp.hpp"
#include "read-formatted-data-table-csv.hpp"
#include "data-table.hpp"
#include "activation-functions.hpp"
#include "derivatives-activation-functions.hpp"
#include "loss-function.hpp"
#include "derivatives-loss-function.hpp"

namespace mlp {
    using NetValuesType = NodeMLP::InputWeightOutputT;
    using NodesCountInLayersT = std::vector<size_t>;
    using TensorT = std::vector<NodeMLP::InputWeightOutputT>;

    // TODO: Variable for undone derivatives
    constexpr NetValuesType stub_var = 1.0;
    
    // In BrainWave, the default learning rate is 0.25 and the default momentum parameter is 0.9.
    // Hence the default value of weight decay in fastai is actually 0.01

    // deafult 0.25
    constexpr double default_learning_rate{ 0.25 };
    //constexpr double default_learning_rate{ 0.45 };
    //constexpr double default_learning_rate{ 0.8 };
    // default 0.01
    constexpr double default_decay{ 0.01 };
    //constexpr double default_decay{ 0.0001 };
    //constexpr double default_decay{ 0.001 };

    // Coefficient of inertia or Momentum. Is needed in learning process to decrease huge changes in purpose function
    // эффекта инерции (momentum)
    // Коэффициент инерции  определяет меру влияния предыдущих подстроек на текущую и, как правило, выбирается исходя из условия
    // 0 < default_momentum < 1
    // if default_momentum = 0, vanilla gradient algorithm will work
    // Default momentum must be 0.9
    constexpr double default_momentum{ 0.9 };
    //constexpr double default_momentum{ 0.5 };
    //constexpr double default_momentum{ 0.35 };
    //constexpr double default_momentum{ 0.1 };
    //constexpr double default_momentum{ 0.0 };
    // http://jre.cplire.ru/win/aug12/4/text.html


    // Difference between accurate value and current value in percents
    constexpr double default_permissible_prediction_error{ 0.0 };

    // Sigmoidal function differes very low in 0 & 1. So network will loss possability to learn if you use 0 or 1 value
    constexpr double kTargetMin{ 0.01 };
    constexpr double kTargetMax{ 0.99 };

    // Randomization interval

    // Interval [-1; 1]
    constexpr double default_weight_randomization{ 1 };
    constexpr double weight_small_min{ 0.001 };
    constexpr double weight_small_max{ 0.1 };


    constexpr NodeMLP::InputWeightOutputT default_bias_output{ 1.0 };

    constexpr char kSavedMLP_Folder[]{ "./nets/" };
    constexpr char kInputDatabasesFolder[]{ "./databases/" };
    constexpr char kInputDataF_Extension[]{ ".csv" };
    constexpr char kMLP_F_Extension[]{ ".mlp" };

// Forward Declaration Section
    // For weights change calculations
    template<typename FuncT> FuncT* const DerivFuncPtr;
    
// !Forward Declaration Section

    //https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/
    //Regression Problem
    //    A problem where you predict a real - value quantity.
    //
    //    Output Layer Configuration : One node with a linear activation unit.
    //    Loss Function : Mean Squared Error(MSE).
    //Binary Classification Problem
    //    A problem where you classify an example as belonging to one of two classes.
    //
    //    The problem is framed as predicting the likelihood of an example belonging to class one, 
    //    e.g.the class that you assign the integer value 1, whereas the other class is assigned the value 0.
    //
    //    Output Layer Configuration : One node with a sigmoid activation unit.
    //    Loss Function : Cross - Entropy, also referred to as Logarithmic loss.
    //Multi - Class Classification Problem
    //    A problem where you classify an example as belonging to one of more than two classes.
    //
    //    The problem is framed as predicting the likelihood of an example belonging to each class.
    //
    //    Output Layer Configuration : One node for each class using the softmax activation function.
    //    Loss Function : Cross - Entropy, also referred to as Logarithmic loss.

    // Popular Algorithms
    // Supervised Learning: Linear Regression, Logistic Regression, Support Vector Machine, K Nearest Neighbour, Random Forest
    // Unsupervised Learing: K-Means, Apriori, C-Means
    // Reinforcement Learning: Q-Learning, SARSA(state action reward(punishment) state action)

    // Tasks
    // Supervised Learning: Regression, Classification. Labeled input, labeled output
    // Unsupervised Learing: Classtering, Associations. Input data.
    // Reinforcement Learning: Games, Robots. Input depends on output.

    /** Classic Multilayer Perceptron */
    class MultiLayerPerceptron {
    public:

        // Type of values in ANN
        using InputTargetPairT = std::pair<NodeMLP::InputWeightOutputT, NodeMLP::InputWeightOutputT>;
        using InputTargetVecT = std::vector<InputTargetPairT>;

        using OneLayerT = std::vector<NodeMLP>;
        using AllNodesLayersT = std::vector<OneLayerT>;

        using DeltaWeightMatrixT = std::vector<std::vector<TensorT>>;

        using EpochIndexT = unsigned long long;


        // All types of Topologies
        // https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464
        enum class TopologyType : unsigned int {
            Perceptron = 0,
            Feed_Forward,
            RBF_Radial_Basis_Function,
            DFF_Deep_Feed_Forward,
            Recurrent_Neural_Networks,
            LSTM_Long_Short_Term_Memory,
            GRU_Gated_Recurrent_Unit,
            AE_Auto_Encoder,
            VAE_Variational_AE,
            DAE_Denoising_AE,
            SAE_Sparse_AE,
            MC_Markov_Chain,
            HN_Hopfield_Networks,
            BM_Boltzmann_Machine,
            RBM_Restricted_BM,
            DBN_Deep_Belief_Network,
            DCN_Deep_Convolutional_Network,
            DN_Deconvolutional_Network,
            DCIGN_Deep_Convolutional_Inverse_Graphics_Network,
            GAN_Generative_Adversarial_Network,
            LSM_Liquid_State_Machine,
            ELM_Extreme_Learning_Machine,
            ESN_Echo_State_Network,
            DRN_Deep_Residual_Network,
            KN_Kohonen_Network,
            SVM_Support_Vector_Machine,
            NTM_Neural_Turing_Machine

            //Feedforward network
            //    single layer network
            //    multilayer network
            //Feedback network
            //    recurrent network
            //        fully recurrent network
            //    Jordan network
        };

        enum class LossFunctionType {
            // Regression Loss Functions
            Mean_Squared_Error = 0,
            Mean_Error,
            Squared_Error,
            Half_Squared_Error,
            Mean_Squared_Logarithmic_Error,
            Root_Mean_Squared_Error,
            Mean_Absolute_Error,
            Poison,

            // Binary Classification Loss Functions
            Binary_Cross_Entropy,
            Hinge_Loss,
            Squared_Hinge,

            // Multi-Class Single Label Classification Loss Functions
            Categorical_Crossentropy,
            Kullback_Leibler_Divergence_Forward,
            Kullback_Leibler_Divergence_Backward,
            Focal_Loss,
            Huber_Loss
            // Likelihood function

            // Multi-Class Multi Label Classification Loss Functions
        };

        enum class TaskType : unsigned int {
            // https://en.wikipedia.org/wiki/Artificial_neural_network
            Classification = 0,
            Regression

            //1. Function approximation, or regression analysis, including time series prediction, fitness approximationand modeling.
            //2. Classification, including patternand sequence recognition, novelty detectionand sequential decision making.[83]
            //3. Data processing, including filtering, clustering, blind source separationand compression.
            //4. Robotics, including directing manipulatorsand prostheses.
        };

        enum class AdaptiveGradientDescentAlgorithms {
            Adagrad = 0,
            Adadelta,
            RMSprop,
            Adam
        };

        // Methods of changing Learning rate by time in process of learning
        enum class LearningRateSchedule {
            Time_based = 0,
            Step_based,
            Exponential,
            Constant
        };


        struct StructureMLP {
            // service variable for linking to bias node
            // In Edge input_ptr is not const, so bias_output is not const
            NodeMLP::InputWeightOutputT bias_output{ default_bias_output };

            // Input values of neural network. It is not neurons of network.
            // nodes_layers_[0] is First Input Layer of network.
            // output of input layer without bias node output
            TensorT output_of_input_layer{};

            /*TensorT convolutional_layers{};
            TensorT encoding_layers{};
            TensorT decoding_layers{};*/

            // Layers of nodes depends from topology of neural network. Start from second layer of neural network
            // Calculation of neural network is going from second layer to the last
            // nodes_layers_[0] is First Input Layer of network. There is no calculation here, only weights.
            // nodes_layers_[0] is empty
            AllNodesLayersT nodes_layers_n_edges{};
        };


        //MultiLayerPerceptron() = default;

        // Create neural network with custom design
        MultiLayerPerceptron(const std::vector<size_t>& layers_dimension_p,
                             NodeMLP::ActivationFunctionType hidden_layer_func_type_p,
                             NodeMLP::ActivationFunctionType output_layer_func_type_p,
                             LossFunctionType loss_func_type_p = LossFunctionType::Mean_Squared_Error,
                             LearningRateSchedule learning_rate_schedule_p = LearningRateSchedule::Time_based,
                             //const double permissible_prediction_error_p = default_permissible_prediction_error,
                             const bool to_normalize_data_p = true,
                             const bool has_bias_p = true);

        // X = W * I; X = output of layer befor activation function;
        // W - weight matrix of layer; I - input_ptr vector of layer
        // Output = func_activation(X); output of layer
        void ForwardPropagateMatrix();
        // Calculate the output value of ALL neural network for input_ptr values
        inline void ForwardPropagateMatrix(const TensorT& input_ptr) {
            SetInputMLP(input_ptr);
            ForwardPropagateMatrix();
        }
        void ProcessOutputOfLayerMtx(size_t layer);

        // Calculate the output value of ALL neural network for input values
        void ForwardPropagateNode();
        // Calculate the output value of ALL neural network for input values
        inline void ForwardPropagateNode(const TensorT& input_ptr) {
            SetInputMLP(input_ptr);
            ForwardPropagateNode();
        }

        void InitLearnMatrixMLP();
        void SupervisedLearnMatrixMLP(const EpochIndexT iter_count, const TensorT& target);
        inline void SupervisedLearnMatrixMLP(const EpochIndexT iter_count, const TensorT& input_ptr,
            const TensorT& target) {
            SetInputMLP(input_ptr);
            SupervisedLearnMatrixMLP(iter_count, target);
        }
        // Когда ошибка равномерно распределяется обратно от выходов в соответствии с вкладом весов в ошибку
        // Веса с большим вкладом больше меняются. Распределение весов: W1 / (W1+W2)
        // f(Sum(w[correct]*i)) = Y+e; w[correct] - это вес, который мы ищемю; i - это вход в узел; Y - ошибочное значение выхода с узла;
        // e - ошибка, распределенная на данный узел с выходов нейронной сети
        // Ошибка[скрытый] = W[транспонированная][скрытый_выходной] * Ошибка[выходной] - матричная запись
        // Calculate Error from last layer to previous
        void BackPropagationOfError();
        // Тарик Рашид Создаем нейронную сеть стр. 97


        // 1) Initialize weight to random small value
        // 2) Repeat Iteration number of steps:
        // 1. Process Network with net input_ptr x
        // 2. Correction(delt = correction) for all outputs (o = output):
        // delt[k] = - o[k]*(1 - o[k])*(t[k] - o[k])
        // 3. delt[j] = o[j]*(1 - o[j])* Sum[k in childer(j)](delt[k]w[j,k])
        // 4. For Bias Node: delt[j] = 1 * Sum[k in childer(j)](delt[k]w[j,k]); derivative of identity function of bias node = 1
        // 3) For all links in neural network
        // delta_w[i,j](n) = default_momentum*delta[i,j](n-1) + (1-default_momentum)*Nu*delt[j]*o[i]
        // w[i,j](n) = w[i,j](n-1) - delta_w[i,j](n)
        // Epochs indexes start from 0

        //template<typename... ColumnT>
        //void SupervisedLearnNodeMLP(const DataTable<ColumnT...>& database, const EpochIndexT max_epoch_index_p) {
        //    if (target[0].size() != layers_dimension_.back()) { throw ErrorRuntimeMLP("Error in dimension of target vector in SupervisedLearnNodeMLP."); }
        //    /*if (input_vec_p.size() != target.size()) {
        //        throw ErrorRuntimeMLP("Different dimensions of target vector & input vector in SupervisedLearnNodeMLP.");
        //    }*/

        //    // Starts new process of network learning
        //    current_epoch_ = 0;
        //    max_epoch_index_ = max_epoch_index_p;
        //    size_t input_data_count{ input_vec_p.size() };
        //    ForwardPropagateNode(); // for first CheckPredictionError
        //    //while (!CheckPredictionError(true) && current_epoch_ < max_epoch_index_) {
        //    for (; current_epoch_ < max_epoch_index_; ++current_epoch_) {
        //        CalcLearningRate();
        //        for (const DataTable<ColumnT...>::RowType& row : database.data_rows) { // Iterate learning input data from row
        //            target_ = target[row];
        //            // TODO: Scale input & target data
        //            ForwardPropagateNode();
        //            CalcCorrectionMatrix();
        //            CalcWeights();
        //        }
        //    }
        //}


        //Making min & max vectors for normalization of input values
        void FindMinMax(const std::vector<TensorT>& input_vec_p,
            std::vector<NodeMLP::InputWeightOutputT>& min_input_p,
            std::vector<NodeMLP::InputWeightOutputT>& max_input_p);

        // Scale input to range from 0 to 1
        void NormalizeVector(std::vector<TensorT>& vec);

        // If to_normalize_data_ = true, normalize data by formula
        // Normalized data = (x - min) / (max - min)
        inline void NormalizeData(std::vector<TensorT>& input_vec_p,
            std::vector<TensorT>& target_p) {
            if (to_normalize_data_) {
                NormalizeVector(input_vec_p);
                NormalizeVector(target_p);
            }
        }

        inline void ResizeTensorInVector(std::vector<TensorT>& vec_of_tensors, const size_t size) {
            for (size_t i = 0, imax = vec_of_tensors.size(); i < imax; ++i) {
                vec_of_tensors[i].resize(size);
            }
        }

        // Converting imported database to input target vectors
        template<size_t kColumnCount, typename ColumnT = float>
        void ConvertDatabaseToInputTarget(const DataTableArray<kColumnCount, ColumnT>& database_p, std::vector<TensorT>& input_p, 
                                          std::vector<TensorT>& target_p) {
            const size_t database_data_rows_count{ database_p.data_rows.size() };
            const size_t input_tensor_size{ layers_dimension_[0] };
            const size_t target_tensor_size{ layers_dimension_.back() };
            input_p.resize(database_data_rows_count);
            target_p.resize(database_data_rows_count);
            ResizeTensorInVector(input_p, input_tensor_size);
            ResizeTensorInVector(target_p, target_tensor_size);
            const size_t input_target_tensor_size{ input_tensor_size + target_tensor_size };
            if (input_target_tensor_size == kColumnCount) { // In database there is all input & target columns
                for (size_t i = 0; i < database_data_rows_count; ++i) {
                    auto front_input_iter{ database_p.data_rows[i].cbegin() };
                    auto last_input_iter{ front_input_iter + input_tensor_size };
                    input_p[i] = TensorT(front_input_iter, last_input_iter);
                    target_p[i] = TensorT(last_input_iter, database_p.data_rows[i].cend());
                }
            }
            else if ((input_target_tensor_size - 1) == kColumnCount && target_tensor_size == 2) { // There is only one target column
                // Binary Classification. Second Possability = 1 - First Possability
                for (size_t i = 0; i < database_data_rows_count; ++i) {
                    auto front_input_iter{ database_p.data_rows[i].cbegin() };
                    input_p[i] = TensorT(front_input_iter, front_input_iter + input_tensor_size);
                    target_p[i][0] = database_p.data_rows[i].back();
                    target_p[i][1] = 1 - target_p[i][0];
                }
            }
            else { throw mlp::ErrorRuntimeMLP("Error in dimension of database, input, target, tensors in ConvertDatabaseToInputTarget."); }
        }
        /*if (input.size() == 0) { throw ErrorRuntimeMLP("Input vector is empty in SupervisedLearnNodeMLP."); }
            if (target.size() == 0) { throw ErrorRuntimeMLP("Target vector is empty in SupervisedLearnNodeMLP."); }
            if (input.size() != target.size()) {
                throw ErrorRuntimeMLP("Different dimensions of target vector & input vector in SupervisedLearnNodeMLP."); }
            if (input[0].size() != GetFirstLayerSize()) {
                throw ErrorRuntimeMLP("Error in dimension of input Tensor in SupervisedLearnNodeMLP."); }
            if (target[0].size() != GetLastLayerSize()) {
                throw ErrorRuntimeMLP("Error in dimension of target Tensor in SupervisedLearnNodeMLP."); }*/


        //template<size_t kColumnCount, typename ColumnT = float>
        //void SupervisedLearnNodeMLP(DataTableArray<kColumnCount, ColumnT>& database,
        //                            const EpochIndexT max_epoch_index_p, bool is_check_erorr_threshold) {
        //    if (input.size() == 0) { throw ErrorRuntimeMLP("Input vector is empty in SupervisedLearnNodeMLP."); }
        //    if (target.size() == 0) { throw ErrorRuntimeMLP("Target vector is empty in SupervisedLearnNodeMLP."); }
        //    if (input.size() != target.size()) {
        //        throw ErrorRuntimeMLP("Different dimensions of target vector & input vector in SupervisedLearnNodeMLP."); }
        //    if (input[0].size() != GetFirstLayerSize()) { 
        //        throw ErrorRuntimeMLP("Error in dimension of input Tensor in SupervisedLearnNodeMLP."); }
        //    if (target[0].size() != GetLastLayerSize()) { 
        //        throw ErrorRuntimeMLP("Error in dimension of target Tensor in SupervisedLearnNodeMLP."); }
        // 
        //    InitLearningProcessNode(input);   // Starts new process of network learning
        //    std::chrono::steady_clock::time_point start{};
        //    std::chrono::steady_clock::time_point end{};
        //    size_t tick_count{};
        //    max_epoch_index_ = max_epoch_index_p;
        //    size_t input_data_count{ input.size() };  // rows count
        //
        //    NormalizeData(input, target);
        //
        //    target_ = target[0];
        //    struct_mlp_.output_of_input_layer = input[0];
        //    ForwardPropagateNode(); // for first CheckPredictionError
        //    while (!to_check_prediction_error && current_epoch_ <= max_epoch_index_
        //        || to_check_prediction_error && !CheckPredictionError() && current_epoch_ <= max_epoch_index_) { // Iterate epochs
        //        if (to_show_learning_process_info) {
        //            WriteEpochInfoInConsole();
        //            if (current_epoch_ == 1) { start = std::chrono::steady_clock::now(); }
        //            else {
        //                std::cout << "Time left: " << (max_epoch_index_ - current_epoch_ + 1) * tick_count * 1e-9 << " seconds\n";
        //                if (!loss_fn_average_history_.empty()) {
        //                    std::cout << "Loss Function = " << loss_fn_average_history_.back() << '\n';
        //                }
        //            }
        //        }
        //
        //        CalcLearningRate();
        //        for (size_t row = 0; row < input_data_count; ++row) { // Iterate learning input data
        //        // TODO: 20% Can be optimized
        //            struct_mlp_.output_of_input_layer = input[row];
        //            target_ = target[row];      // !20%
        //            ForwardPropagateNode();     // 21%
        //            CalcCorrectionMatrix();     // 40%  big nodes count -> 30%
        //            CalcWeights();              // 27%  big nodes count -> 70%
        //        // 11%
        //            if (to_save_loss_fn_mean_history_ || !to_save_loss_fn_mean_history_ && current_epoch_ == max_epoch_index_) {
        //                loss_fn_values_[row] = CalcLossFunction(GetOutputTensor(), target[row]);
        //            }
        //            // !11%
        //        }
        //
        //        if (to_save_loss_fn_mean_history_ || !to_save_loss_fn_mean_history_ && current_epoch_ == max_epoch_index_) { // Save history
        //            // Mean, average of all loss values from input data set
        //            loss_fn_average_history_.emplace_back(std::accumulate(loss_fn_values_.begin(),
        //                loss_fn_values_.end(), static_cast<NetValuesType>(0)) / loss_fn_values_.size());
        //        }
        //        if (to_show_learning_process_info && current_epoch_ == 1) { // Show learning info
        //            end = std::chrono::steady_clock::now();
        //            tick_count = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        //        }
        //        ++current_epoch_;
        //    }
        //    CoutPredictionErrorPercent();
        //}

        // 1) Initialize weight to random small value
        // 2) Repeat Iteration number of steps:
        // 1. Process Network with net input_ptr x
        // 2. Correction(delt = correction) for all outputs (o = output):
        // delt[k] = - o[k]*(1 - o[k])*(t[k] - o[k])
        // 3. delt[j] = o[j]*(1 - o[j])* Sum[k in childer(j)](delt[k]w[j,k])
        // 4. For Bias Node: delt[j] = 1 * Sum[k in childer(j)](delt[k]w[j,k]); derivative of identity function of bias node = 1
        // 3) For all links in neural network
        // delta_w[i,j](n) = default_momentum*delta[i,j](n-1) + (1-default_momentum)*Nu*delt[j]*o[i]
        // w[i,j](n) = w[i,j](n-1) - delta_w[i,j](n)
        // Epochs indexes start from 0
        // Input & target data can be changed by normalization operation
        void SupervisedLearnNodeMLP(std::vector<TensorT>& input_vec_p, std::vector<TensorT>& target,
                                    const EpochIndexT max_epoch_index_p, bool is_check_erorr_threshold = false);
        // https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%BE%D0%B1%D1%80%D0%B0%D1%82%D0%BD%D0%BE%D0%B3%D0%BE_%D1%80%D0%B0%D1%81%D0%BF%D1%80%D0%BE%D1%81%D1%82%D1%80%D0%B0%D0%BD%D0%B5%D0%BD%D0%B8%D1%8F_%D0%BE%D1%88%D0%B8%D0%B1%D0%BA%D0%B8
        // Learning algorithm will work until it reaches erorrs_threshold errors measure


        // Generalization error is a measure of how accurately an algorithm is able to predict outcome values for previously unseen data
        // https://en.wikipedia.org/wiki/Generalization_error
        void CalcGeneralizationError();
        // Correctness, Error measure, performance of neural network
        // Метод наименьших квадратов. Cумма квадратов расстояний от выходных сигналов сети до их требуемых значений
        // E({w[ij]}) = 1/2*Sum[k in output2]((t[k]-o[k])^2)
        // t[k] = target value of output layer
        void AnalyzeMLP();
        // https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%BE%D0%B1%D1%80%D0%B0%D1%82%D0%BD%D0%BE%D0%B3%D0%BE_%D1%80%D0%B0%D1%81%D0%BF%D1%80%D0%BE%D1%81%D1%82%D1%80%D0%B0%D0%BD%D0%B5%D0%BD%D0%B8%D1%8F_%D0%BE%D1%88%D0%B8%D0%B1%D0%BA%D0%B8
        // Функция потерь. https://ru.wikipedia.org/wiki/%D0%A4%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D1%8F_%D0%BF%D0%BE%D1%82%D0%B5%D1%80%D1%8C

        // True if error target - output <= mean_absolute_error_percent_ percents for each output
        // If continuation of learning will reduce quality of results you need to stop learning process
        bool CheckPredictionError();
        // For matrix form of learning process
        bool CheckPredictionErrorMatrix();

        // Needed in the beggining of learnning process in some learning algorighms
        void RandomizeWeightsMLP(double min = 0.0001, double max = 20.0);
        // Randomization corrected by count of weights linked to node
        // formula: 1 / sqrt(edges_count)
        void RandomizeAllWeightsByNodesCountMLP();





        // Load neural network from file
        // file_name without full path & without extension
        void LoadMLP(const std::string& file_name);

        // Save neural network's weights, topology, activation functions to file
        // file_name without full path & without extension
        void SaveMLP(const std::string& file_name);


        TensorT GetInputMLP() const { return struct_mlp_.output_of_input_layer; };
        inline const AllNodesLayersT& GetNodesLayers() const { return struct_mlp_.nodes_layers_n_edges; };
        inline const size_t GetLayersCount() const { return struct_mlp_.nodes_layers_n_edges.size(); };
        // Get count of nodes in layer
        inline const size_t GetLayerSize(const size_t layer) const { return struct_mlp_.nodes_layers_n_edges[layer].size(); };
        inline const TopologyType GetTopology() const { return topology_; };
        inline const NodeMLP::ActivationFunctionType GetHiddenLayerActivationFunction() const { return hidden_layer_actv_func_type_; };
        inline const NodeMLP::ActivationFunctionType GetOutputLayerActivationFunction() const { return output_layer_actv_func_type_; };
        inline const double GetHasBias() const { return has_bias_; };
        inline const double GetLearningRate() const { return learning_rate_; };
        inline const double GetInitialLearningRate() const { return learning_rate_initial_; };
        inline const double GetDecay() const { return decay_; };
        // Needed for StepBased learning rate schedule algorithm
        inline const double GetLearningEpochsDropRate() const { return epochs_drop_rate_; };
        inline const TensorT& GetTarget() const { return target_; };
        inline const double GetToCheckPredictionError() const { return to_check_prediction_error; };
        inline const double GetPermissiblePredictionError() const { return permissible_prediction_error_; };
        inline const std::vector<double>& GetPredictionErrorPercent_() const { return prediction_error_percent_; };
        inline const double GetIsErrorForEachOutput() const { return is_error_for_each_output; };
        inline const std::vector<NetValuesType>& GetLossFnMeanHistory() const { return loss_fn_average_history_; };
        inline const std::vector<NetValuesType>& GetLossFnValues() const { return loss_fn_values_; };
        inline const bool GetFlagSaveLossFnMeanHistory() const { return to_save_loss_fn_mean_history_; };
        inline const bool GetToShowLearningProcInfo() const { return to_show_learning_process_info; };


        inline const std::vector<Eigen::VectorXd>& GetOutputOfLayersMtx() const { return output_of_layers_; }
        inline const Eigen::VectorXd& GetOutputOfLayerVec(size_t index) const {
            if (index > 0) { return output_of_layers_[index]; }
            else { throw ErrorRuntimeMLP("Index of vector must be > than 0"); }
        }

        inline const OneLayerT& GetLastLayer() const { return struct_mlp_.nodes_layers_n_edges.back(); }
        inline const size_t GetLastLayerIndex() const { return struct_mlp_.nodes_layers_n_edges.size() - 1; }
        inline const size_t GetLastLayerSize() const { return struct_mlp_.nodes_layers_n_edges.back().size(); }
        inline const size_t GetFirstLayerSize() const { return struct_mlp_.output_of_input_layer.size(); }
        inline const NodeMLP& GetLastNodeInLayer(const size_t layer) const { return struct_mlp_.nodes_layers_n_edges[layer].back(); };
        inline const TensorT GetOutputTensor() const {
            auto back{ struct_mlp_.nodes_layers_n_edges.back() };
            TensorT output_vec{};
            output_vec.reserve(back.size());
            for (const NodeMLP& node : back) {
                output_vec.emplace_back(node.GetOutput());
            }
            return output_vec;
        }
        inline const bool GetToNormalizeData() { return to_normalize_data_; }


        inline void SetToCheckPredictionError(const bool flag) {
            to_check_prediction_error = flag;
            ResizePredictionErrorPercent();
        };
        inline void SetPermissiblePredictionError(const double value) { permissible_prediction_error_ = value; };
        inline void SetIsErrorForEachOutput(const bool flag) { 
            is_error_for_each_output = flag;
            ResizePredictionErrorPercent();
        };
        inline void SetCheckingErrorConfig(const bool to_check_error, const double permissible_error, const bool error_for_each) {
            to_check_prediction_error = to_check_error;
            permissible_prediction_error_ = permissible_error;
            is_error_for_each_output = error_for_each;
            ResizePredictionErrorPercent();
        };
        inline void SetFlagSaveLossFnMeanHistory(const bool flag) { to_save_loss_fn_mean_history_ = flag; };
        inline void SetToShowLearningProcInfo(const bool flag) { to_show_learning_process_info = flag; };
        inline void SetInputMLP(const TensorT& input_mlp_p) {
            if (input_mlp_p.size() != layers_dimension_[0]) { throw ErrorRuntimeMLP("Wrong dimension of input data vector."); }
            struct_mlp_.output_of_input_layer = input_mlp_p;
            //ConnectSecondLayerToInput(); // Reconnect first layer
        };
        inline void SetToNormalizeData(bool to_normalize_data_p) { to_normalize_data_ = to_normalize_data_p; }
        // Parametr needed for learning rate schedule StepBased algorithm
        inline void SetDropRate(double drop_rate) { epochs_drop_rate_ = drop_rate; }

        inline void ResetLearningRate() { learning_rate_ = learning_rate_initial_; };


        // Helper functions

                // Processing and Displaying one input vector. Node Form of processing.
        void ProcessNDisplayOneInputMtrx(const std::vector<NodeMLP::InputWeightOutputT>& input_vec);
        // Processing and Displaying many input vectors. Node Form of processing.
        void ProcessNDisplayAllInputMtrx(const std::vector<std::vector<NodeMLP::InputWeightOutputT>>& all_input_vec);

        // Processing and Displaying one input vector. Node Form of processing.
        void ProcessNDisplayOneInput(const std::vector<NodeMLP::InputWeightOutputT>& input_vec);
        // Processing and Displaying many input vectors. Node Form of processing.
        void ProcessNDisplayAllInput(const std::vector<std::vector<NodeMLP::InputWeightOutputT>>& all_input_vec);
        // Helper functions


    private:
        // Gets weight between node i of layer-1 & node j of layer. j is children(closer to output of network) of i
        // i, j starts from 0
        inline const NodeMLP::InputWeightOutputT GetWeightIJ(const size_t layer_j, const size_t i, const size_t j) const {
            return struct_mlp_.nodes_layers_n_edges[layer_j][j].GetWeight(i);
        }
        inline void SetWeightIJ(const size_t layer_j, const size_t i, const size_t j, const NodeMLP::InputWeightOutputT value) {
            return struct_mlp_.nodes_layers_n_edges[layer_j][j].SetWeight(i, value);
        }

        // Volatile variant of getting last node in layer
        inline NodeMLP& GetLastNodeInLayerV(const size_t layer) { return *std::prev(struct_mlp_.nodes_layers_n_edges[layer].end()); };
        // Volatile reference to layer
        inline OneLayerT& GetLayerV(const size_t index) { return struct_mlp_.nodes_layers_n_edges[index]; }
        // Volatile type of GetLastLayer func
        inline OneLayerT& GetLastLayerV() { return struct_mlp_.nodes_layers_n_edges[GetLastLayerIndex()]; }

        inline NetValuesType CalcActivationFunction(NodeMLP::ActivationFunctionType func_act_type_p,
            const NetValuesType x) {
            switch (func_act_type_p) {
            case NodeMLP::ActivationFunctionType::Identity:                                 return fn_actv::IdentityFn(x);
            case NodeMLP::ActivationFunctionType::Binary_step:                              return fn_actv::BinaryStepFn(x);
            case NodeMLP::ActivationFunctionType::Sigmoid_Logistic_soft_step:               return fn_actv::SigmoidLogisticFn(x);
            case NodeMLP::ActivationFunctionType::Hyperbolic_tangent:                       return fn_actv::HyperbolicTangentFn(x);
            case NodeMLP::ActivationFunctionType::ReLU_Rectified_linear_unit:               return fn_actv::RectifiedLinearUnitFn(x);
            case NodeMLP::ActivationFunctionType::GELU_Gaussian_Error_Linear_Unit:          return fn_actv::GaussianErrorLinearUnitFn(x);
            case NodeMLP::ActivationFunctionType::Softplus:                                 return fn_actv::SoftplusFn(x);
            case NodeMLP::ActivationFunctionType::ELU_Exponential_linear_unit:              return fn_actv::ExponentialLinearUnitFn(x, stub_var);
            case NodeMLP::ActivationFunctionType::SELU_Scaled_exponential_linear_unit:      return fn_actv::ScaledExponentialLinearUnitFn(x);
            case NodeMLP::ActivationFunctionType::Leaky_ReLU_Leaky_rectified_linear_unit:   return fn_actv::LeakyRectifiedLinearUnitFn(x);
            case NodeMLP::ActivationFunctionType::PReLU_Parameteric_rectified_linear_unit:  return fn_actv::ParametricRectifiedLinearUnitFn(x, stub_var);
            case NodeMLP::ActivationFunctionType::SiLU_Sigmoid_linear_unit:                 return fn_actv::SigmoidLinearUnitFn(x);
            case NodeMLP::ActivationFunctionType::Mish:                                     return fn_actv::MishFn(x);
            case NodeMLP::ActivationFunctionType::Gaussian:                                 return fn_actv::GaussianFn(x);
            case NodeMLP::ActivationFunctionType::GrowingCosineUnitFn:                      return fn_actv::GrowingCosineUnitFn(x);
                //case NodeMLP::ActivationFunctionType::Softmax:                                  return fn_actv::SoftmaxFn(x);
                //case NodeMLP::ActivationFunctionType::Maxout:                                   return fn_actv::MaxoutFn(x);
            case NodeMLP::ActivationFunctionType::Linear:                                   return fn_actv::LinearFn(x, stub_var, stub_var);
            case NodeMLP::ActivationFunctionType::GaussianRBFFn:                            return fn_actv::GaussianRBFFn(x, stub_var, stub_var);
            case NodeMLP::ActivationFunctionType::HeavisideFn:                              return fn_actv::HeavisideFn(x, stub_var, stub_var);
            case NodeMLP::ActivationFunctionType::MultiquadraticsFn:                        return fn_actv::MultiquadraticsFn(x, stub_var, stub_var);
            case NodeMLP::ActivationFunctionType::SwishFn:                                  return fn_actv::SwishFn(x);
            case NodeMLP::ActivationFunctionType::HardSigmoidFn:                            return fn_actv::HardSigmoidFn(x);
            }
        }

        // df(S) / dS | S = S[j]    ; S[j] = Sum[all i] w[i,j] * x[i]
        inline NetValuesType CalcActivationFunctionDeriv(NodeMLP::ActivationFunctionType func_act_type_p,
            const NetValuesType x) {
            switch (func_act_type_p) {
            case NodeMLP::ActivationFunctionType::Identity:                                 return fn_deriv::IdentityFn_deriv();
            case NodeMLP::ActivationFunctionType::Binary_step:                              return fn_deriv::BinaryStepFn_deriv(x);
            case NodeMLP::ActivationFunctionType::Sigmoid_Logistic_soft_step:               return fn_deriv::SigmoidLogisticFn_deriv(x);
            case NodeMLP::ActivationFunctionType::Hyperbolic_tangent:                       return fn_deriv::HyperbolicTangentFn_deriv(x);
            case NodeMLP::ActivationFunctionType::ReLU_Rectified_linear_unit:               return fn_deriv::RectifiedLinearUnitFn_deriv(x);
            case NodeMLP::ActivationFunctionType::GELU_Gaussian_Error_Linear_Unit:          return fn_deriv::GaussianErrorLinearUnitFn_deriv(stub_var, x, stub_var);
            case NodeMLP::ActivationFunctionType::Softplus:                                 return fn_deriv::SoftplusFn_deriv(x);
            case NodeMLP::ActivationFunctionType::ELU_Exponential_linear_unit:              return fn_deriv::ExponentialLinearUnitFn_deriv(x, stub_var);
            case NodeMLP::ActivationFunctionType::SELU_Scaled_exponential_linear_unit:      return fn_deriv::ScaledExponentialLinearUnitFn_deriv(x);
            case NodeMLP::ActivationFunctionType::Leaky_ReLU_Leaky_rectified_linear_unit:   return fn_deriv::LeakyRectifiedLinearUnitFn_deriv(x);
            case NodeMLP::ActivationFunctionType::PReLU_Parameteric_rectified_linear_unit:  return fn_deriv::ParametricRectifiedLinearUnitFn_deriv(x, stub_var);
            case NodeMLP::ActivationFunctionType::SiLU_Sigmoid_linear_unit:                 return fn_deriv::SigmoidLinearUnitFn_deriv(x);
            case NodeMLP::ActivationFunctionType::Mish:                                     return fn_deriv::MishFn_deriv(x);
            case NodeMLP::ActivationFunctionType::Gaussian:                                 return fn_deriv::GaussianFn_deriv(x);
            case NodeMLP::ActivationFunctionType::GrowingCosineUnitFn:                      return fn_deriv::GrowingCosineUnitFn_deriv(x);
                //case NodeMLP::ActivationFunctionType::Softmax:                                  return fn_deriv::SoftmaxFn_deriv(x);
                //case NodeMLP::ActivationFunctionType::Maxout:                                   return fn_deriv::MaxoutFn_deriv(x);
            case NodeMLP::ActivationFunctionType::Linear:                                   return fn_deriv::LinearFn_deriv(stub_var);
                //case NodeMLP::ActivationFunctionType::GaussianRBFFn:                            return fn_deriv::GaussianRBFFn_deriv(x);
            case NodeMLP::ActivationFunctionType::HeavisideFn:                              return fn_deriv::HeavisideFn_deriv(x, stub_var);
                //case NodeMLP::ActivationFunctionType::MultiquadraticsFn:                        return fn_deriv::MultiquadraticsFn_deriv(x);
            case NodeMLP::ActivationFunctionType::SwishFn:                                  return fn_deriv::SwishFn_deriv(x, stub_var);
                //case NodeMLP::ActivationFunctionType::HardSigmoidFn:                            return fn_deriv::HardSigmoidFn_deriv(x);
            default:                                                                        return fn_deriv::RectifiedLinearUnitFn_deriv(x);
            }
        }

        // Calculate loss function vector on all input data & mean loss function to measure the accuracy of network
        // Evaluate the accuracy of the learned function. After parameter adjustment and learning, the performance of the resulting function should be measured on a test set that is separate from the training set.
        inline NetValuesType CalcLossFunction(const std::vector<NetValuesType>& output,
            const std::vector<NetValuesType>& target) {
            switch (loss_func_type_) {
                // Regression Loss Functions
            case LossFunctionType::Mean_Error:                      return fn_loss::MeanError(output, target);
            case LossFunctionType::Squared_Error:                   return fn_loss::SquaredError(output, target);
            case LossFunctionType::Half_Squared_Error:              return fn_loss::HalfSquaredError(output, target);
            case LossFunctionType::Mean_Squared_Error:              return fn_loss::MeanSquaredError(output, target);
            case LossFunctionType::Mean_Squared_Logarithmic_Error:  return fn_loss::MeanSquaredLogarithmicError(output, target);
            case LossFunctionType::Root_Mean_Squared_Error:         return fn_loss::RootMeanSquaredError(output, target);
            case LossFunctionType::Mean_Absolute_Error:             return fn_loss::MeanAbsoluteError(output, target);
            case LossFunctionType::Poison:                          return fn_loss::Poison(output, target);

                // Binary Classification Loss Functions
            case LossFunctionType::Binary_Cross_Entropy:            return fn_loss::BinaryCrossentropy(output, target);
            case LossFunctionType::Hinge_Loss:                      return fn_loss::HingeLoss(output, target);
            case LossFunctionType::Squared_Hinge:                   return fn_loss::SquaredHinge(output, target);

                // Multi-Class Classification Loss Functions
            case LossFunctionType::Categorical_Crossentropy:        return fn_loss::CategoricalCrossentropy(output, target);
            case LossFunctionType::
                Kullback_Leibler_Divergence_Forward:                return fn_loss::KullbackLieblerDivergenceForward(output, target);
            case LossFunctionType::
                Kullback_Leibler_Divergence_Backward:               return fn_loss::KullbackLieblerDivergenceBackward(output, target);
            case LossFunctionType::Focal_Loss:                      return fn_loss::FocalLoss(output, target, 1);
            case LossFunctionType::Huber_Loss:                      return fn_loss::HuberLoss(output, target, 1);
            }
        }

        //inline NetValuesType CalcLossFunctionDeriv(const std::vector<NetValuesType>& output, const std::vector<NetValuesType>& target) {
        // dE / do[j]
        inline NetValuesType CalcLossFunctionDerivRespectOutputJ(const NetValuesType output, const NetValuesType target,
                                                                 const size_t k_max = 0) {
            switch (loss_func_type_) {
            // Regression Loss Functions
            case LossFunctionType::Mean_Squared_Error:              return fn_loss_deriv::MeanSquaredError_deriv(output, target, k_max);
            case LossFunctionType::Mean_Error:                      return fn_loss_deriv::MeanError_deriv(k_max);
            case LossFunctionType::Squared_Error:                   return fn_loss_deriv::SquaredError_deriv(output, target);
            case LossFunctionType::Half_Squared_Error:              return fn_loss_deriv::HalfSquaredError_deriv(output, target);
            case LossFunctionType::Mean_Squared_Logarithmic_Error:  return fn_loss_deriv::MeanSquaredLogarithmicError_deriv(output, target, k_max);
            //case LossFunctionType::Root_Mean_Squared_Error:         return fn_loss_deriv::RootMeanSquaredError_deriv(output, target);
            case LossFunctionType::Mean_Absolute_Error:             return fn_loss_deriv::MeanAbsoluteError_deriv(output, target, k_max);
            case LossFunctionType::Poison:                          return fn_loss_deriv::Poison_deriv(output, target, k_max);

            // Binary Classification Loss Functions
            case LossFunctionType::Binary_Cross_Entropy:            return fn_loss_deriv::BinaryCrossentropy_deriv(output, target, k_max);
            //case LossFunctionType::Hinge_Loss:                      return fn_loss_deriv::HingeLoss_deriv(output, target);
            //case LossFunctionType::Squared_Hinge:                   return fn_loss_deriv::SquaredHinge_deriv(output, target);

            // Multi-Class Classification Loss Functions
            //case LossFunctionType::Categorical_Crossentropy:        return fn_loss_deriv::CategoricalCrossentropy_deriv(output, target);
            //case LossFunctionType::Kullback_Leibler_Divergence:     return fn_loss_deriv::KullbackLieblerDivergence_deriv(output, target);
            //case LossFunctionType::Kullback_Leibler_Divergence:     return fn_loss_deriv::KullbackLieblerDivergence_deriv(output, target);
            //case LossFunctionType::Focal_Loss:                      return fn_loss_deriv::FocalLoss_deriv(output, target, 1);
            //case LossFunctionType::Huber_Loss:                      return fn_loss_deriv::HuberLoss_deriv(output, target, 1);
            }
        }

        void SetAllActivationFuncs(NodeMLP::ActivationFunctionType hidden_layer_func_type_p,
            NodeMLP::ActivationFunctionType output_layer_func_type_p);
        void SetAllNodesIndexes();

        void CalcPredictionErrorPercent();
        void CoutPredictionErrorPercent();

        // Connects all node links except first layer & input_ptr
        void CreateAllEdges();
        // Connect second layer = nodes_layers_[0]
        void ConnectSecondLayerToInput();

        // Initialization for learning process & network processing
        inline void InitLearningProcessNode(std::vector<TensorT>& input_vec) {
            ResetLearningRate();
            ResetDeltaWeights();
            current_epoch_ = 0;
            loss_fn_values_.resize(input_vec.size());
        }
        void Initialization();
        void InitWeightsMatrix();
        void InitInputVector();
        void InitTargetVector();
        void InitOutputMatrix();
        void InitErrorMatrix();
        //void InitLearnMatrixMLP();

        // Sets all weights of all neurons to default & set output_ready indicator to false
        void ResetMLP();
        inline void ResetDeltaWeights() {
            for (std::vector<TensorT>& delta_weight_of_layer : delta_weights_) {
                for (TensorT& tensor : delta_weight_of_layer) {
                    for (NodeMLP::InputWeightOutputT& tensor_elem : tensor) {
                        tensor_elem = 0;
                    }
                }
            }

        }
        void ResizeLayersNInput();
        inline void ResizePredictionErrorPercent() {
            if (to_check_prediction_error && is_error_for_each_output) { prediction_error_percent_.resize(GetLastLayerSize()); }
            else if (!to_check_prediction_error
                     || to_check_prediction_error && !is_error_for_each_output) { prediction_error_percent_.resize(1); }
        }

        // Resize Correction matrix to nodes_layers_ dimension
        // Correction Matrix consists of delta correction for each node of network
        void InitCorrectionMtx();
        // Resize Delta weights matrix to nodes_layers_ dimension
        void InitDeltaWeightsMtx();
        // Init all neccessary variable for node learning mlp
        void InitLearnNodeMLP();

        // Calculate correction for all nodes for backpropagation method
        void CalcCorrectionMatrix();
        // Calculate weights of neural network
        void CalcWeights();
        // For matrix form of neural network learning
        // new W[jk] = old_W[jk] - Nu * E[k] * O[k]*(1 - O[k]) * O[j][transpont]
        void CalcWeightsMatrixForm();
        void SaveWeightsFrmMatrixToNodes();

        NodeMLP::InputWeightOutputT DerivativeLossFunction() const;
        constexpr NodeMLP::InputWeightOutputT DerivativeActivationFunction() const;

        // Write in console epoch info
        inline void WriteEpochInfoInConsole() {
    // Critical to performance code section. Don't use cls.
            //system("cls"); 
            std::cout << "Current Epoch index = " << current_epoch_;
            std::cout << "\nMax epoch index = " << max_epoch_index_ << '\n';
    // !Critical to performance code section
        }

//!SupervisedLearnNodeMLP Rate Schedule====================================================

// https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
//https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/

        // Calculates learning rate using choosen method
        void CalcLearningRate();

        // When learning rate is constant
        void ConstantLRate();

        // Time-based Learning rate schedule
        // Nu[n+1] = Nu[n] / (1 + d*n)
        // Nu - Learning rate
        // d - decay parametr
        // n - iteration step
        void TimeBasedLRateSchedule();

        // Step-based Learning rate schedule
        // Nu[n] = Nu[0]*d^(floor((1 + n)/r))
        // Nu[n] - learning rate at iteration n
        // Nu[0] - the initial learning rate
        // d - how much the learning rate should change at each drop
        // r - corresponds to the droprate, or how often the rate should be dropped (10 corresponds to a drop every 10 iterations)
        // floor - The floor function here drops the value of its input_ptr to 0 for all values smaller than 1.
        // Drop rate is number of epochs from wich learning rate drops to half
        void StepBasedLRateSchedule();

        // Exponential Learning rate schedule
        // Nu[n] = Nu[0]*exp^(-d*n)
        // Nu[n] - learning rate at iteration n
        // Nu[0] - the initial learning rate
        // d - decay parametr
        void ExponentialLRateSchedule();

        void AdaptiveLRate();

//!SupervisedLearnNodeMLP Rate Schedule====================================================

// Helper functions

        // Rescaling of the data from the original range so that all values are within the range of 0 and 1.
        // y = (x - min) / (max - min)
        inline NodeMLP::InputWeightOutputT NormalizeValue(const NodeMLP::InputWeightOutputT x, const NodeMLP::InputWeightOutputT min, const NodeMLP::InputWeightOutputT max) {
            return (x - min) / (max - min);
        }
        // https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/

        // Scale input_ptr to be in interval of -[1;1] for sigmoid activation function
        // input_ptr*a/b
        void ScaleInput(TensorT& input_ptr, NodeMLP::InputWeightOutputT a = 1.0, NodeMLP::InputWeightOutputT b = 100.0);
        inline double Scale(NodeMLP::InputWeightOutputT value, NodeMLP::InputWeightOutputT a = 1.0, NodeMLP::InputWeightOutputT b = 100.0) {
            return value * a / b;
        }

// !Helper functions

        // Structure of all network
        StructureMLP struct_mlp_{};

        // Vector of Target values
        TensorT target_{};
        
        TopologyType topology_{ TopologyType::Feed_Forward };

        // Activation function of hidden layer
        NodeMLP::ActivationFunctionType hidden_layer_actv_func_type_{ NodeMLP::ActivationFunctionType::SiLU_Sigmoid_linear_unit };
        // Activation function of output layer
        NodeMLP::ActivationFunctionType output_layer_actv_func_type_{ NodeMLP::ActivationFunctionType::Softmax };
        // Loss function for network weights changing & calc of network accuracy
        LossFunctionType loss_func_type_{ LossFunctionType::Mean_Squared_Error };

        // Type of algorithm of changing learning rate throught epochs
        LearningRateSchedule learning_rate_schedule_{ LearningRateSchedule::Time_based };

        // Bias works like shift in output function.
        // All layers except the last have one bias node.
        // Bias will be in input_n_weights_ of nodes of all layer except the last layer.
        // But bias value is used in every layer except first layer
        // Bias is neuron with one const input_ptr = 1, identity y_in=x function activation and with evaluated weight of output
        bool has_bias_{ true };

        // Use normalization of data when learn & process network
        bool to_normalize_data_{ true };

        // Learning Rate at the first iteration step. Can be changed when loading ann.
        double learning_rate_initial_{ default_learning_rate };

        // A learning rate schedule changes the learning rate during learningand is most often changed between epochs / iterations
        // https://en.wikipedia.org/wiki/Learning_rate
        double learning_rate_{ learning_rate_initial_ };

        // Needed in calculation learning_rate_
        // All the data from database learned from one epoch. Then epoch increased, decreasing learning rate.
        // Network learning speed is very high at the beginning of the learning process.
        // Then it makes little corrections at high epoch value
        // Epochs start from 0.
        EpochIndexT current_epoch_{ 0 };
        // Max index of epoch during last learning process
        EpochIndexT max_epoch_index_{ 0 };

        // Decay is needed for changing learning rate
        double decay_{ default_decay };

        // Momentum is needed to prevent wrong results of gradient algoritm
        double momentum_{ default_momentum };

        // If flag is false, no checks are making
        bool to_check_prediction_error{ false };
        // Value in percents indicating when neural network will stop learning
        // Default value is 5%
        // Calculation formula: mean absolute error loss function =  1/n * Sum|target - prediction|
        double permissible_prediction_error_{ default_permissible_prediction_error };
        // Mean absolute loss error function value on all output or on each output
        std::vector<double> prediction_error_percent_{};
        // Calculate prediction_error_percent_ for all outputs together or for each
        bool is_error_for_each_output{ false };
        // Loss function values on all of input data set of last learning epoch
        std::vector<NodeMLP::InputWeightOutputT> loss_fn_values_{};
        // element = Loss function value average of all rows in one epoch. History on all epochs of learning on input data set. This helps to track results of net learning.
        std::vector<NodeMLP::InputWeightOutputT> loss_fn_average_history_{};
        // Flag of saving history of mean loss function values
        bool to_save_loss_fn_mean_history_{ false };
        bool to_show_learning_process_info{ false };

        // Parameter for StepBased learning rate schedule
        // Number of epochs from wich learning rate drops to half
        double epochs_drop_rate_{ 2 };


        // Поправка к узлам
        // delt[k](k in output) = derivative_Activation_Function_of_output_layer * derivative_Loss_Funciton = dE / dNet_sum
        // delt[j](j in hidden) = derivative_Activation_Function_of_hidden_layer * Sum[k in childern](delt[k] * weight[j,k])
        std::vector<TensorT> correction_matrix_{};
        // look wikipedia

        // Changing of weights on current iteration
        // delta_weights_[0] = delta of weights between First layer and Second layer
        // delta_weights_[i] = delta of weights between i layer and i+1 layer
        // Structure of delta_weights_ is equal to structure of nodes_layers_n_edges in struct_mlp_
        DeltaWeightMatrixT delta_weights_{};

// Matrix data

        //Eigen::VectorXd input_vec_{};
        Eigen::VectorXd target_vec_{};
        // All weights matrixes include weights of First input_ptr layer. Used in matrix calculation of network output
        // w11  w21 w[i,j]
        // w12  w22
        // Rows are all weights connected to neuron i in layer
        // Starts from first layer
        // w[i,j]: i - neurons in previous layer; j - neuron in current layer
        // columns count = number of neurons in previous layer
        // rows count = number of neurons in current layer
        std::vector<Eigen::MatrixXd> weights_in_layer_{};
        // Starts from first layer
        std::vector<Eigen::VectorXd> output_of_layers_{};
        std::vector<Eigen::VectorXd> error_in_layers_{};

// Service data

        // Service variable for softmax output layer activation function
        std::vector<NodeMLP::InputWeightOutputT> output_layer_net_input_vec_{};
        std::vector<size_t> layers_dimension_{};


        // ANN Name - user defined 
        //std::string ann_name_{};

        // Decay. Decay serves to settle the learning in a nice place and avoid oscillations, a situation that may arise when a too high constant learning rate makes the learning jump back and forth over a minimum, and is controlled by a hyperparameter.
        // Momentum. Momentum is analogous to a ball rolling down a hill; we want the ball to settle at the lowest point of the hill 
        //
        // Adaptive learning rate
        // The issue with learning rate schedules is that they all depend on hyperparameters that must be manually chosen for each given learning session and may vary greatly depending on the problem at hand or the model used.
        // Stochastic gradient descent. Adagrad, Adadelta, RMSprop, and Adam
        // https://en.wikipedia.org/wiki/Stochastic_gradient_descent#AdaGrad


        /*topology_;
        task_;
        learning_type;
        learning_rules_;*/
    };

    
// Helper functions

    // Load input_ptr data for neural network from csv file excel table
    // DataTableTuple
    template<typename... ColumnT>
    DataTableTuple<ColumnT...> LoadInputDataFrmCSV(const std::string& file_name) {
        const std::string file_path{ kInputDatabasesFolder + file_name + kInputDataF_Extension };
        return file::ReadFormattedDataTableCSV<ColumnT...>(file_path);
    };

    // Load input_ptr data for neural network from csv file excel table
    // DataTableArray
    template<size_t kColumnCount, typename ColumnT>
    DataTableArray<kColumnCount, ColumnT> LoadInputDataFrmCSV(const std::string& file_name) {
        const std::string file_path{ kInputDatabasesFolder + file_name + kInputDataF_Extension };
        return file::ReadFormattedDataTableCSV<kColumnCount, ColumnT>(file_path);
    };

// Rules for seting hidden layer

    // Ns / (alpha * (Ni + No))
    // Ni = number of input nodes
    // No = number of nodes in output layer
    // Ns = number of samples in training data set
    // α = an arbitrary scaling factor usually 2 - 10
    inline size_t NodesInHiddenLayer(const size_t input_count, const size_t output_count,
                                     const size_t samples_count, const size_t alpha = 2.0) {
        return samples_count / (alpha * (input_count + output_count));
    }

    //number of hidden layers equals one; and (ii) the number of neurons in that layer is the mean of the neurons in the input and output layers.
    inline size_t NodesInHiddenLayerMeanRule(const size_t input_count, const size_t output_count) {
        return (input_count + output_count) / 2;
    }
// !Rules for seting hidden layer
    
// !Helper functions

}

#endif // !MULTILAYER_PERCEPTRON_HPP_
