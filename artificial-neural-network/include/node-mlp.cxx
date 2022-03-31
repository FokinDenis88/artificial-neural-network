#ifndef NODE_MLP_CXX_
#define NODE_MLP_CXX_

module;

#include <vector>
#include <utility>

export module node_mlp;

import activation_functions;
import errors_mlp;

export namespace mlp {
    constexpr double default_weight{ 0.01 };

    // Multilayer Perceptron Neuron = NodeMLP
    // Y = F(yin)
    // yin = summ from i to n(xi * wi)
    class NodeMLP {
    //template <typename valueT = double>
    public:
        friend class MultiLayerPerceptron;
        //friend void MultiLayerPerceptron::CalcCorrectionMatrix();
        //friend void MultiLayerPerceptron::CalcWeights();

        // type of Input, Weight & Output
        using InputWeightOutputT = long double;

        // Pair of input_ptr & weight
        /*using InputWeightPairT = std::pair<const InputWeightOutputT*, InputWeightOutputT>;
        using InputWeightVecT = std::vector<InputWeightPairT>;*/

        enum class ActivationFunctionType : unsigned int {
            //The following table compares the properties of several activation functions that are functions of one fold x from the previous layer or layers:
            Identity = 0,                                   // IdentityFn(T yin)
            Binary_step,                                    // BinaryStepFn(T yin)
            Sigmoid_Logistic_soft_step,                     // SigmoidLogisticFn(T yin) {
            Hyperbolic_tangent,                             // HyperbolicTangentFn(T yin)
            ReLU_Rectified_linear_unit,                     // RectifiedLinearUnitFn(T yin)
            GELU_Gaussian_Error_Linear_Unit,                // GaussianErrorLinearUnitFn(T yin)
            Softplus,                                       // SoftplusFn(T yin)
            ELU_Exponential_linear_unit,                    // ExponentialLinearUnitFn(T a, T yin)
            SELU_Scaled_exponential_linear_unit,            // ScaledExponentialLinearUnitFn(T yin)
            Leaky_ReLU_Leaky_rectified_linear_unit,         // LeakyRectifiedLinearUnitFn(T yin)
            PReLU_Parameteric_rectified_linear_unit,        // ParametricRectifiedLinearUnitFn(T a, T yin)
            SiLU_Sigmoid_linear_unit,                       //Sigmoid shrinkage, SiL, Swish-‍1   // SigmoidLinearUnitFn(T yin)
            Mish,                                           // MishFn(T yin)
            Gaussian,                                       // GaussianFn(T yin)
            GrowingCosineUnitFn,                            // Growing Cosine Unit (GCU)

            //The following table lists activation functions that are not functions of a single fold x from the previous layer or layers:
            Softmax,                                        // exp(x[i]) / (Sum[j=1 to J] exp(xj)) for i=1 to J
            Maxout,                                         // MaxoutFn(std::vector<T> yin)
            Linear,                                         // LinearFn(T a, T yin, T b)
            GaussianRBFFn,                                  // GaussianRBFFn(T y_in, T c, T sigma)
            HeavisideFn,                                    // HeavisideFn(T y_in, T a, T b)
            MultiquadraticsFn,                              // MultiquadraticsFn(T y_in, T c, T a)
            SwishFn,                                        // SwishFn(const T x)
            HardSigmoidFn,                                  // HardSigmoidFn(const T x)

            NoActivationFunction // For empty layer
        };
        // https://en.wikipedia.org/wiki/Activation_function
        //1. Ridge activation functions
        //Linear
        //ReLU
        //Heaviside
        //Logistic(Sigmoid: Binary sigmoidal function, Bipolar sigmoidal function)
        //
        //2. Radial activation functions
        //    Gaussian
        //    Multiquadratics
        //
        //3. Folding activation functions

        enum class LearningType : unsigned int {
            supervised = 0,
            unsupervised,
            reinforcement,
            self
        };

        enum class LearningRulesType : unsigned int {
            Hebbian = 0,
            Correlation,
            Perceptron,
            Delta_or_Widrow_Hoff,
            Competitive_or_Winner_Takes_All,
            Outstar

            // https://en.wikipedia.org/wiki/Learning_rule
            //1. Hebbian - Neocognitron, Brain - State - in - a - Box(BSB)
            //2. Gradient Descent - ADALINE, Hopfield Network, Recurrent Neural Network
            //3. Competitive - Learning Vector Quantisation, Self - Organising Feature Map, Adaptive Resonance Theory
            //4. Stochastic - Boltzmann Machine, Cauchy Machine
        };

        // Edge, link between two nodes
        struct Edge {
            // Inputs are links to values. Input of second, last, current node.
            // Input like Dendrites
            // Output of previous nodes
            InputWeightOutputT* input_ptr{};
            // Weights like Synapses in Neuron
            // Often small weight sizes are used such as 0.1 or 0.01 or smaller. Weights from tesflow.
            InputWeightOutputT weight{};

            // The start node of the edge-link
            NodeMLP* edge_start_node_ptr{};
            // edge_end_node_ptr is the current node;

            // Input and Weights. Inputs are links to values.
            // Input like Dendrites & Weights like Synapses in Neuron
            // Pair help to make correct data
            // Bias is stored in last element
            //InputWeightVecT input_n_weights_{};
        };
        using Edges = std::vector<Edge>;

        // Calculate Summ of All input * weights
        inline InputWeightOutputT CalcNetInput() {
            net_input_ = 0;
            for (const Edge& edge : edges_) {
                net_input_ += (*edge.input_ptr) * edge.weight;
            }
            return net_input_;
        }

        // Evaluate output of Neuron
        // Calculate Y of node
        void ProcessNodeOutput();


        // Needed in the beggining of learnning process in some learning algorighms
        void RandomizeWeightsNode(double min = 0.0001, double max = 0.0);

        // Randomization corrected by count of weights linked to node
        // formula: 1 / sqrt(edges_count)
        void RandomizeWeightsByNodesCount();

        // Sets all weights of all neurons to default & set output_ready indicator to false
        void ResetNode();

        // Add Links one node output of other node to input_ptr of current node
        inline void AddEdgeToNode(NodeMLP& node_start, const InputWeightOutputT weight = default_weight) {
            edges_.emplace_back(node_start.GetOutputAddress(), weight, &node_start);
        }
        // Add Links outputs of other nodes to input_ptr of current node
        inline void AddEdgeToNode(std::vector<NodeMLP>& nodes_for_link) {
            if (!nodes_for_link.empty()) {
                for (NodeMLP& node : nodes_for_link) {
                    edges_.emplace_back(node.GetOutputAddress(), default_weight, &node);
                }
            }
        };
        // Add Links outputs of other nodes to input_ptr of current node
        inline void AddEdgeToNode(std::vector<NodeMLP>& nodes_for_link, const std::vector<InputWeightOutputT>& weights) {
            if (!nodes_for_link.empty()) {
                if (weights.empty()) { throw ErrorRuntimeMLP("There is no weights for nodes to add edges"); }
                for (int i = 0, imax = nodes_for_link.size(); i < imax; ++i) {
                    edges_.emplace_back(nodes_for_link[i].GetOutputAddress(), weights[i], &nodes_for_link[i]);
                }
            }
        };
        // Add Links of input_ptr of first layer to input_ptr of neural network
        inline void AddEdgeToInput(InputWeightOutputT* const input_ptr, const InputWeightOutputT weight = default_weight) {
            edges_.emplace_back(input_ptr, weight, nullptr);
        }
        // Add Link to Bias or to output value of node in previous layer
        inline void AddEdgeToBias(InputWeightOutputT* const bias_output, const InputWeightOutputT weight = default_weight) {
            edges_.emplace_back(bias_output, weight, nullptr);
        }
        
        // Delete all links between current node and other nodes in neural network
        inline void DeleteAllEdges() { edges_.clear(); };
        // ResetLinks Clears all pairs in input_weights, and sets weights to default values


        inline const Edges& GetEdges() const { return edges_; };
        inline const size_t GetEdgesCount() const { return edges_.size(); };
        // Index is not a node number in previous layer. It is index in vector of all weights of nodes
        inline const InputWeightOutputT GetWeight(const long index) const {
            // High performance cost
            //if (index >= edges_.size()) { throw ErrorRuntimeMLP("There is no such weight. index is out of range in GetWeight."); }
            return edges_[index].weight; 
        };
        inline const std::vector<InputWeightOutputT> GetWeights() const {
            std::vector<InputWeightOutputT> weights;
            weights.reserve(edges_.size());
            for (const Edge& edge : edges_) {
                weights.emplace_back(edge.weight);
            }
            return weights;
        };
        inline ActivationFunctionType GetActivationFunctionType() const { return activation_function_type_; };
        inline const InputWeightOutputT GetOutput() const { return output_; };
        
        inline const size_t GetIndex() const { return index_; };


        inline void SetEdges(const Edges& edges_p) { edges_ = edges_p; };
        inline void SetEdges(const long index, const Edge& edge_p) { edges_[index] = edge_p; };
        inline void SetWeight(const long index, const InputWeightOutputT weight) { 
            if (index >= edges_.size()) { throw ErrorRuntimeMLP("There is no such weight. index is out of range in GetWeight."); }
            edges_[index].weight = weight; 
        };
        /*inline void SetWeights(const std::vector<double> weights) {
            input_n_weights_.resize(weights.size());
        };*/
        inline void SetActivationFunctionType(ActivationFunctionType func_type) { activation_function_type_ = func_type; };
        inline void SetIndex(size_t value) { index_ = value; };
        // Not to use except Softmax Activation Function
        inline void SetSoftmaxResults(const InputWeightOutputT result) {
            // Needs check, but performance is critical
            // if (activation_function_type_ == ActivationFunctionType::Softmax) {}
            output_ = result;
        }

        inline void ResizeEdgesVec(const size_t size) { 
            size_t old_size{ edges_.size() };
            edges_.resize(size); //Clear old links
            for (long i = old_size; i < size; ++i) { // set weights of new links to defauls; i = old_size - index of next new elem
                edges_[i].weight = default_weight; 
            }
        };


        [[deprecated]]
        inline void SupervisedLearnNodeMLP(const double learning_rate_p, const InputWeightOutputT target_output) {
            //DeltaLR(learning_rate_p, target_output);
        };
        // Run SupervisedLearnNodeMLP algorithm for node
        [[deprecated]]
        inline void LearnNode() {
            //HebbianLR();
        };
        [[deprecated]]
        inline void LearnNode(const double learning_rate_p, const InputWeightOutputT target_output) {
            //OutstarLR(learning_rate_p, target_output);
            DeltaLR(learning_rate_p, target_output);
            //HebbianLR(learning_rate_p);
            //PerceptronLR(learning_rate_p, target_output);
        };

    private:
        // Service function for adding edges in node
        inline InputWeightOutputT* const GetOutputAddress() { return &output_; };

//SupervisedLearnNodeMLP Neuron====================================================

    // Unsupervised Learning

        // Hebbian Learning Rule
        // Cells that fire together wire together.
        // When an axon of cell A is near enough to excite a cell B and repeatedly or persistently takes part in firing it, some growth process or metabolic change takes place in one or both cells such that A’s efficiency, as one of the cells firing B, is increased.
        // wi(t) = Nu * xi(t) * Y(t)
        // wi - weight of i input_ptr to current neuron
        // Nu - Learning rate
        // xi - i input_ptr to current neuron
        // Y = output of current neuron
        // 2 variant of formula: wij = xi*xj
        // weights updated after every training example
        // At the start, values of all weights are set to zero. 
        // no reflexive connections allowed
        // Unstable learning rule
        // Type of Learning = Unsupervised Learning
        // Function activation = Any
        void HebbianLR(const double learning_rate_p);
        // https://en.wikipedia.org/wiki/Hebbian_theory
        // https://en.wikipedia.org/wiki/Learning_rule
        // https://www.tutorialspoint.com/artificial_neural_network/artificial_neural_network_learning_adaptation.htm
        // https://data-flair.training/blogs/learning-rules-in-neural-network/#:~:text=Learning%20rule%20or%20Learning%20process,in%20a%20specific%20data%20environment.
        // https://en.wikipedia.org/wiki/Oja%27s_rule   Good formula

        // It is a modification of the standard Hebb's Rule. It is demonstrably stable, unlike Hebb's rule.
        // Oja's rule defines the change in presynaptic weights w given the output response y of a neuron to its inputs x to be:
        // delta w = w[n+1]  - w[n] = Nu*y[n](x[n] - y[n]*w[n])
        // Stable learning rule
        // Type of Learning = Unsupervised Learning
        // Function activation = Any
        void OjaLR();
        // https://en.wikipedia.org/wiki/Oja%27s_rule

        // Competitive or Winner Takes All Learning Rule
        // This network is just like a single layer feedforward network with feedback connection between outputs. The competitors never support themselves.
        // The output unit with the highest activation to a given input_ptr pattern, will be declared the winner.
        // If any neuron, say yk⁡, wants to win, then its induced local field theoutputofsummationunit, say vk, must be the largest among all the other neurons in the network.
        // Condition to be a winner:
        // yk = 1; vk > vj for all j
        // yk = 0; otherwise
        // yk - neuron; vk - local field theoutputofsummationunit induced by yk
        // Condition of sum total of weight: the sum total of weights to a particular output neuron is going to be 1
        // sumj(wkj) = 1; for all k
        // Change of weight for winner:
        // delta_wkj = -Nu*(xj - wkj); if k wins
        // delta_wkj = 0 ; if k losses
        // Type of Learning = Unsupervised Learning
        // Function activation = ?
        void CompetitiveLR(const double learning_rate_p);
        // https://www.tutorialspoint.com/artificial_neural_network/artificial_neural_network_learning_adaptation.htm

    // Supervised Learning

        // Weights between responding neurons should be more positive, and weights between neurons with opposite reaction should be more negative.
        // delta_wij = Nu*xi*tj
        // tj is the desired value of output signal
        // This training algorithm usually starts with the initialization of weights to zero.
        // Type of Learning = Supervised Learning
        // Function activation = ?
        void CorrelationLR(const double learning_rate_p, const InputWeightOutputT target_output);
        // https://data-flair.training/blogs/learning-rules-in-neural-network/#:~:text=Learning%20rule%20or%20Learning%20process,in%20a%20specific%20data%20environment.

        // This rule is an error correcting the supervised learning algorithm of single layer feedforward networks with linear activation function, introduced by Rosenblatt.
        // w(new) = w(old) + Nu*(t-Y)*x
        // Error is restricted to having values of 0, 1, or -1
        // Type of Learning = Supervised Learning
        // Function activation = BinaryStep = Threshold output function
        void PerceptronLR(const double learning_rate_p, const InputWeightOutputT target_output);

        // Widrow−HoffRule or Delta Learning Rule. Also called Least Mean Square LMS.
        // The base of this rule is gradient-descent approach, which continues forever.
        // Delta rule updates the synaptic weights so as to minimize the net input_ptr to the output unit and the target value.
        // delta_wi = Nu * xi * ej
        // delta_wi - weight change for i'th ⁡pattern
        // Nu - the positive & constant Learning rate
        // xi - the input_ptr value from pre-synaptic neuron
        // ej = (t - Y) the difference between the desired/target output and the actual output ⁡yin; ej = error = delta
        // Same as PerceptronLR(), but with any activation function
        // Type of Learning = Supervised Learning
        // Function activation = Continuous activation function
        void DeltaLR(const double learning_rate_p, const InputWeightOutputT target_output);

        // This rule is applied over the neurons arranged in a layer. It is specially designed to produce a desired output d of the layer of p neurons.
        // delta_wj = Nu * (t - wj)
        // t - desired output of the layer
        // Type of Learning = Supervised Learning
        // Function activation = Any activation function
        void OutstarLR(const double learning_rate_p, const InputWeightOutputT target_output);
        //2. Gradient Descent - ADALINE, Hopfield Network, Recurrent Neural Network
        //4. Stochastic - Boltzmann Machine, Cauchy Machine

        // Generalized Delta Learning Rule or Back Propagation Algorithm
        // The process by which a Multi Layer Perceptron learns is called the Backpropagation algorithm
        void BackPropagationAlgorithm();

    
    // Reinforcement

        void DeepReinforcementLearning();
        // BEST FOR GAMES

    // Undefined category algorithms

        // Neural network with fuzzy layer. Such algorithms are used in youtube
        void LearnFuzzyANN();

        void XorGateLR();
        void MeanSquaredError();


//!SupervisedLearnNodeMLP Neuron====================================================

        // Bias is stored in last element
        Edges edges_{};

        ActivationFunctionType activation_function_type_;

        // Like Axon in Neuron
        InputWeightOutputT output_{};

        // Sum of all input * weights from edges
        InputWeightOutputT net_input_{ 0 };

        // Index in vector layer of neuron
        size_t index_{};
    };
}

#endif // !NODE_MLP_CXX_
