#include "networks/main_network.hpp"

#include <iomanip> // setprecision
#include <iostream>
#include <chrono>
#include <exception>
#include <stdexcept>

#include "read-formatted-data-table-csv.hpp"

#include "multilayer-perceptron.hpp"

//#include "loss_function.h"

namespace mlp {
    namespace main_network {
        int Run() {
            try {
#define NODEFORM
//#define MATRIXFORM

                auto start{ std::chrono::steady_clock::now() };
                auto end{ std::chrono::steady_clock::now() };
                auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

                //mlp::MultiLayerPerceptron my_ann(mlp::NodesCountInLayersT{ 1, 100, 300, 500, 700, 3 }, 
                mlp::MultiLayerPerceptron my_ann(mlp::NodesCountInLayersT{ 1, 10, 100, 1 },
                                                mlp::NodeMLP::ActivationFunctionType::Sigmoid_Logistic_soft_step,
                                                mlp::NodeMLP::ActivationFunctionType::Sigmoid_Logistic_soft_step,
                                                5, mlp::MultiLayerPerceptron::LearningRateSchedule::Time_based);

                constexpr long long iterations_count{ 100 };
                std::cout << std::setprecision(20);

                
                //mlp::TensorT letters{ 'a' };
                mlp::TensorT letters{ 'a', 'b', 'c', 'd', 'e', 'f', 'g' };
                mlp::TensorT numbers{ '1', '2', '3', '4', '5', '6', '7', '8' };
                mlp::TensorT others{ '\'', '\"', '+', '=', '-'};

                /*my_ann.ScaleInput(letters);
                my_ann.ScaleInput(numbers);
                my_ann.ScaleInput(others);*/

#ifdef NODEFORM
                start = std::chrono::steady_clock::now();

                for (double input_ptr : letters) {
                    mlp::TensorT target{ mlp::kTargetMax, mlp::kTargetMin, mlp::kTargetMin };
                    mlp::TensorT input_vec{ mlp::TensorT{ input_ptr } };
                    //my_ann.LearnNodeMLP(input_vec, target, iterations_count);
                }

                end = std::chrono::steady_clock::now();
                elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                std::cout << "Elapsed time Learn Neurons by NodeForm: " << elapsed_time << " miliseconds\n\n";
                // 39503

                for (double input_ptr : numbers) {
                    mlp::TensorT target{ mlp::kTargetMin, mlp::kTargetMax, mlp::kTargetMin };
                    mlp::TensorT input_vec{ mlp::TensorT{ input_ptr } };
                    //my_ann.LearnNodeMLP(iterations_count, input_vec, target);
                    //my_ann.LearnNodeMLP(100, target, input_vec);
                }
                for (double input_ptr : others) {
                    mlp::TensorT target{ mlp::kTargetMin, mlp::kTargetMin, mlp::kTargetMax };
                    mlp::TensorT input_vec{ mlp::TensorT{ input_ptr } };
                    //my_ann.LearnNodeMLP(iterations_count, input_vec, target);
                    //my_ann.LearnNodeMLP(700, target, input_vec);
                }


                // Test learning results
                start = std::chrono::steady_clock::now();

                my_ann.ProcessNDisplayOneInput(letters);

                end = std::chrono::steady_clock::now();
                elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                std::cout << "Elapsed time Process Neurons by NodeForm: " << elapsed_time << " miliseconds\n\n";

                // Test of matrix form Processing
                my_ann.InitLearnMatrixMLP();
                start = std::chrono::steady_clock::now();

                my_ann.ProcessNDisplayOneInputMtrx(letters);

                end = std::chrono::steady_clock::now();
                elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                std::cout << "Elapsed time Process Neurons by MatrixForm: " << elapsed_time << " miliseconds\n\n";
                // !Test of matrix form Processing      

                my_ann.ProcessNDisplayOneInput(numbers);
                my_ann.ProcessNDisplayOneInput(others);

#endif // NODEFORM

#ifdef MATRIXFORM
                start = std::chrono::steady_clock::now();

                for (double input_ptr : letters) {
                    mlp::TensorT target{ mlp::kTargetMax, mlp::kTargetMin, mlp::kTargetMin };
                    mlp::TensorT input_vec{ mlp::TensorT{ input_ptr } };
                    my_ann.LearnMatrixMLP(iterations_count, input_vec, target);
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

                mlp::ProcessNDisplayOneInputMtrx(numbers, my_ann);
                mlp::ProcessNDisplayOneInputMtrx(others, my_ann);
#endif // MATRIXFORM

                // Performance Test
                /*auto start1 = std::chrono::steady_clock::now();
                mlp::ProcessNDisplayOneInput(letters, my_ann);
                auto end1 = std::chrono::steady_clock::now();
                auto elapsed_time1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
                std::cout << "Elapsed time ProcessNDisplayOneInput: " << elapsed_time1 << " milliseconds\n";

                auto start2 = std::chrono::steady_clock::now();
                mlp::ProcessNDisplayOneInputMtrx(letters, my_ann);
                auto end2 = std::chrono::steady_clock::now();
                auto elapsed_time2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
                std::cout << "Elapsed time ProcessNDisplayOneInputMtrx: " << elapsed_time2 << " milliseconds\n";

                std::cout << "Elapsed time 1 - 2: " << elapsed_time1 - elapsed_time2 << " milliseconds\n";*/
                //! Performance Test


                constexpr char net_name[]{ "my_first_ann" };
                my_ann.SaveMLP(net_name);
                mlp::MultiLayerPerceptron test_load{};
                test_load.LoadMLP(net_name);

                std::cout << '\n';
                std::cout << '\n';
                std::cout << '\n';

                int a = 0;
            }
            catch (const std::logic_error& error) {
                std::cerr << "Runtime Error: " << error.what();
                return -1;
            }
            catch (const std::runtime_error& error) {
                std::cerr << "Runtime Error: " << error.what();
                return -1;
            }
            catch (...) {
                std::cerr << "Fatal Error";
                return -1;
            }
        }

    }
}
