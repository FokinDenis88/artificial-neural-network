#ifndef BINARY_OR_H
#define BINARY_OR_H

#include <iostream>
#include <chrono>
#include <exception>
#include <stdexcept>

// bad alloc
#include <new>

#include "MultiLayerPerceptron.h"

#include "loss_function.h"

namespace mlp {
    namespace binary_or {

        typedef DataTableArray<3, float> TableBinaryOR;
        typedef DataTableArray<4, float> TableBinaryORFullTarget;

        const mlp::TensorT kTestInput{ 0.4, 0.4 };
        const std::vector<mlp::TensorT> kTestVecInput{ { 25, 70 } };

        int Run() {
            try {
#define NODEFORM_BINARY_OR
//#define MATRIXFORM_BINARY_OR
//#define LOADCUSTOM_BINARY_OR
//#define SAVENETWORK_BINARY_OR

                auto start{ std::chrono::steady_clock::now() };
                auto end{ std::chrono::steady_clock::now() };
                auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 500, 500, 500, 700, 2 }, 
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 400, 400, 400, 400, 2 }, 
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 100, 400, 700, 2 },
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 350, 350, 350, 2 },
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 100000, 2 },
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 100, 200, 2 },
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 100, 2 },
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 1000, 1000, 1000, 2 },
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 10000, 10000, 2 },
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 10000000, 2 },
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 10000, 2 },
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 5000, 5000, 2 },
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 2500, 2500, 2 },
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 2000, 2 },
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 1000, 400, 2 },
                mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 1000, 2 },
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 500, 2 },
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 150, 2 },
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 100, 2 },
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 20, 20, 2 },
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 3, 2 },
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 2 },
                                                 mlp::NodeMLP::ActivationFunctionType::ReLU_Rectified_linear_unit,
                                                 //mlp::NodeMLP::ActivationFunctionType::Sigmoid_Logistic_soft_step,
                                                 //mlp::NodeMLP::ActivationFunctionType::Hyperbolic_tangent,
                                                 
                                                 //mlp::NodeMLP::ActivationFunctionType::NoActivationFunction,
                                                 mlp::NodeMLP::ActivationFunctionType::Softmax,
                                                 //mlp::NodeMLP::ActivationFunctionType::Sigmoid_Logistic_soft_step,
                                                 //mlp::NodeMLP::ActivationFunctionType::ReLU_Rectified_linear_unit,
                                                 //mlp::NodeMLP::ActivationFunctionType::Linear,

                                                 //mlp::MultiLayerPerceptron::LossFunctionType::Half_Squared_Error,
                                                 //mlp::MultiLayerPerceptron::LossFunctionType::Mean_Squared_Error,
                                                 mlp::MultiLayerPerceptron::LossFunctionType::Categorical_Crossentropy,

                                                 mlp::MultiLayerPerceptron::LearningRateSchedule::Exponential
                                                 //mlp::MultiLayerPerceptron::LearningRateSchedule::Time_based
                                                 );

                //constexpr long long max_epoch_index{ 100000 };
                //constexpr long long max_epoch_index{ 10000 };
                //constexpr long long max_epoch_index{ 5000 };
                //constexpr long long max_epoch_index{ 3000 };
                //constexpr long long max_epoch_index{ 1500 };
                constexpr long long max_epoch_index{ 1000 };
                //constexpr long long max_epoch_index{ 500 };
                //constexpr long long max_epoch_index{ 100 };
                //constexpr long long max_epoch_index{ 250 };
                //constexpr long long max_epoch_index{ 100 };
                //constexpr long long max_epoch_index{ 70 };
                //constexpr long long max_epoch_index{ 10 };
                //constexpr long long max_epoch_index{ 1 };
                std::cout << std::setprecision(25);


                DataTableArray<3, float> database{ mlp::LoadInputDataFrmCSV<3, float>("BinaryOR") };
                //DataTableArray<4, float> database{ mlp::LoadInputDataFrmCSV<4, float>("BinaryOR - Full Target") };
                //DataTableArray<3, float> database{ mlp::LoadInputDataFrmCSV<3, float>("BinaryOR_Normalized") };

                /*my_mlp.SetFlagSaveLossFnMeanHistory(true);
                my_mlp.SetToShowLearningProcInfo(true);
                my_mlp.SetCheckingErrorConfig(true, 0.0, true);*/

                my_mlp.SetFlagSaveLossFnMeanHistory(false);
                my_mlp.SetToShowLearningProcInfo(true);
                my_mlp.SetCheckingErrorConfig(false, 0.0, false);

#ifdef NODEFORM_BINARY_OR
                start = std::chrono::steady_clock::now();

                std::vector<mlp::TensorT> or_input{};
                std::vector<mlp::TensorT> or_target{};
                my_mlp.ConvertDatabaseToInputTarget(database, or_input, or_target);
                /*std::vector<mlp::TensorT> or_input    { { 0.6, 0.6 }, { 0.6, 0.4 },  { 0.4, 0.6 }, { 0.4, 0.4 } };
                std::vector<mlp::TensorT> or_target   { { 0.6, 0.4 }, { 0.6, 0.4 },  { 0.6, 0.4 }, { 0.4, 0.6 } };*/

                /*std::vector<mlp::TensorT> or_input  { { 0.9, 0.9 }, { 0.9, 0.1 },  { 0.1, 0.9 }, { 0.1, 0.1 } };
                std::vector<mlp::TensorT> or_target { { 0.9, 0.1 }, { 0.9, 0.1 },  { 0.9, 0.1 }, { 0.1, 0.9 } };*/
                /*const std::vector<mlp::TensorT> or_input  { { 0.99, 0.99 }, { 0.99, 0.01 },  { 0.01, 0.99 }, { 0.01, 0.01 } };
                const std::vector<mlp::TensorT> or_target { { 0.99, 0.01 }, { 0.99, 0.01 },  { 0.99, 0.01 }, { 0.01, 0.99 } };*/
                /*const std::vector<mlp::TensorT> or_input{ { 0.99, 0.99 }, { 0.99, 0.01 },  { 0.01, 0.99 }, { 0.01, 0.01 }, { 0.01, 0.01 }, { 0.01, 0.01 } };
                const std::vector<mlp::TensorT> or_target{ { 0.99, 0.01 }, { 0.99, 0.01 },  { 0.99, 0.01 }, { 0.01, 0.99 }, { 0.01, 0.99 }, { 0.01, 0.99 } };*/
                my_mlp.SupervisedLearnNodeMLP(or_input, or_target, max_epoch_index);

                end = std::chrono::steady_clock::now();
                elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
                std::cout << "Elapsed time Learn Neurons by NodeForm: " << elapsed_time << " seconds = ";
                elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                std::cout << elapsed_time << " miliseconds\n\n";


                // Test learning results
                start = std::chrono::steady_clock::now();

                my_mlp.ProcessNDisplayOneInput(kTestInput);

                end = std::chrono::steady_clock::now();
                elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                std::cout << "\nElapsed time Process Neurons by NodeForm: " << elapsed_time << " milliseconds = ";
                elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
                std::cout << elapsed_time << " nanoseconds\n\n";

                
        // Test of matrix form Processing
                std::cout << '\n';
                my_mlp.InitLearnMatrixMLP();
                start = std::chrono::steady_clock::now();

                my_mlp.ProcessNDisplayOneInputMtrx(kTestInput);

                end = std::chrono::steady_clock::now();
                elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                std::cout << "\nElapsed time Process Neurons by NodeForm: " << elapsed_time << " milliseconds = ";
                elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
                std::cout << elapsed_time << " nanoseconds\n\n";
        // !Test of matrix form Processing      

                // Process all data in base
                std::vector<mlp::TensorT> test_inputs{};
                for (const TableBinaryOR::RowType& row : database.data_rows) {
                    test_inputs.push_back(mlp::TensorT{ static_cast<double>(std::get<0>(row)), 
                                                        static_cast<double>(std::get<1>(row)) });
                }
                /*for (const TableBinaryORFullTarget::RowType& row : database.data_rows) {
                    test_inputs.push_back(mlp::TensorT{ static_cast<double>(std::get<0>(row)),
                                                        static_cast<double>(std::get<1>(row)) });
                }*/
                std::cout << '\n';
                my_mlp.ProcessNDisplayAllInput(test_inputs);
                std::cout << "Process all data in base\n\n";

#endif // NODEFORM

#ifdef MATRIXFORM_BINARY_OR
                start = std::chrono::steady_clock::now();

                for (double input_ptr : letters) {
                    mlp::TensorT target{ mlp::kTargetMax, mlp::kTargetMin, mlp::kTargetMin };
                    mlp::TensorT input_vec{ mlp::TensorT{ input_ptr } };
                    my_ann.SupervisedLearnMatrixMLP(max_epoch_index, input_vec, target);
                }

                end = std::chrono::steady_clock::now();
                elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                std::cout << "Elapsed time Learn Neurons by MatrixForm: " << elapsed_time << " miliseconds\n\n";

                // Test learning results
                start = std::chrono::steady_clock::now();

                mlp::ProcessNDisplayOneInputMtrx(letters, my_ann);

                end = std::chrono::steady_clock::now();
                elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                std::cout << "Elapsed time Process Neurons by MatrixForm: " << elapsed_time << " miliseconds\n\n";
#endif // MATRIXFORM

#ifdef LOADCUSTOM_BINARY_OR
                mlp::MultiLayerPerceptron custom_mlp{};
                custom_mlp.LoadMLP("zero_classifier");
                mlp::ProcessNDisplay(TensorT{ 10000, 2000, 6000, -49999 }, custom_mlp);
#endif // LOADCUSTOM

#ifdef SAVENETWORK_BINARY_OR
                constexpr char net_name[]{ "zero_classifier" };
                my_mlp.SaveMLP(net_name);
                mlp::MultiLayerPerceptron test_load{};
                test_load.LoadMLP(net_name);
#endif // SAVENETWORK


                std::cout << '\n';
                std::cout << '\n';

                int a = 0;
            }
            catch (const ErrorRuntimeMLP& error) {
                std::cerr << "ErrorRuntimeMLP: " << error.what();
                return -1;
            }
            catch (const ErrorLogicMLP& error) {
                std::cerr << "ErrorLogicMLP: " << error.what();
                return -1;
            }
            catch (const ErrorSaveMLP& error) {
                std::cerr << "ErrorRuntimeMLP: " << error.what();
                return -1;
            }
            catch (const ErrorLoadMLP& error) {
                std::cerr << "ErrorLoadMLP: " << error.what();
                return -1;
            }
            catch (const std::logic_error& error) {
                std::cerr << "Logic error: " << error.what();
                return -1;
            }
            catch (const std::runtime_error& error) {
                std::cerr << "Runtime error: " << error.what();
                return -1;
            }
            catch (const std::out_of_range& error) {
                std::cerr << "Out of range error: " << error.what();
                return -1;
            }
            catch (const std::bad_alloc& error) {
                std::cerr << "Bad allocation error: " << error.what();
                return -1;
            }
            catch (...) {
                std::cerr << "Fatal error";
                return -1;
            }
        }

    }
}

#endif // !BINARY_OR_H