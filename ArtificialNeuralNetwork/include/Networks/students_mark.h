#ifndef STUDENTS_MARK_H
#define STUDENTS_MARK_H

#include <iostream>
#include <chrono>
#include <exception>
#include <stdexcept>

#include "MultiLayerPerceptron.h"

#include "loss_function.h"

namespace mlp {
    namespace students_mark {

        const mlp::TensorT kTestInput{ 25, 70 };
        const std::vector<mlp::TensorT> kTestVecInput{ { 25, 70 } };

        int Run() {
            try {
#define NODEFORM_STUDENTS_MARK
//#define MATRIXFORM_STUDENTS_MARK
//#define LOADCUSTOM_STUDENTS_MARK
#define SAVENETWORK_STUDENTS_MARK

                auto start{ std::chrono::steady_clock::now() };
                auto end{ std::chrono::steady_clock::now() };
                auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 1, 100, 300, 500, 700, 3 }, 
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 1, 100, 400, 700, 2 },
                //mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 1, 100, 200, 2 },
                mlp::MultiLayerPerceptron my_mlp(mlp::NodesCountInLayersT{ 2, 100, 300, 500, 700, 2 },
                                                 mlp::NodeMLP::ActivationFunctionType::Sigmoid_Logistic_soft_step,
                                                 mlp::NodeMLP::ActivationFunctionType::Softmax,
                                                 5, mlp::MultiLayerPerceptron::LearningRateSchedule::Time_based);

                constexpr long long max_epoch_index{ 1 };
                std::cout << std::setprecision(20);

                const file::tables::TableStudentsMark database{ mlp::LoadInputDataFrmCSV<int, int, float>("StudentsMark",
                                                                                            file::tables::StudentsMark) };

#ifdef NODEFORM_STUDENTS_MARK
                start = std::chrono::steady_clock::now();

                const file::tables::TableStudentsMark::RowType& zero_row{ database.data_rows[0] };
                const file::tables::TableStudentsMark::RowType& second_row{ database.data_rows[1] };
                constexpr size_t target_column_indx{ 2 };
                constexpr size_t scale_size{ 100 };
                for (int i = 0; i < 1; ++i) {
                    for (const file::tables::TableStudentsMark::RowType& row : database.data_rows) {
                        /*my_mlp.LearnNodeMLP(max_epoch_index, mlp::TensorT{ { mlp::Scale(static_cast<double>(std::get<0>(zero_row), 1, 1000)) } },
                                            mlp::TensorT{ std::get<1>(zero_row), 1 - std::get<1>(zero_row) });*/

                                            /*my_mlp.LearnNodeMLP(max_epoch_index, mlp::TensorT{ { mlp::Scale(static_cast<double>(std::get<0>(second_row), 1, 1000)) } },
                                                                mlp::TensorT{ std::get<1>(second_row) });*/

                        /*mlp::TensorT input_vec{ { mlp::Scale(static_cast<double>(std::get<0>(row)), 1, scale_size),
                                                  mlp::Scale(static_cast<double>(std::get<1>(row)), 1, scale_size)} };*/
                        /*mlp::TensorT input_vec{ { static_cast<double>(std::get<0>(row)),
                                                  static_cast<double>(std::get<1>(row))} };*/
                        mlp::TensorT target{ std::get<target_column_indx>(row), 1 - std::get<target_column_indx>(row) };
                        //my_mlp.LearnNodeMLP(input_vec, target, max_epoch_index);
                    }
                }

                end = std::chrono::steady_clock::now();
                elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                std::cout << "Elapsed time Learn Neurons by NodeForm: " << elapsed_time << " miliseconds\n\n";


                // Test learning results
                start = std::chrono::steady_clock::now();

                my_mlp.ProcessNDisplayOneInput(kTestInput);

                end = std::chrono::steady_clock::now();
                elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                std::cout << "\nElapsed time Process Neurons by NodeForm: " << elapsed_time << " miliseconds\n\n";


        // Test of matrix form Processing
                my_mlp.InitLearnMatrixMLP();
                start = std::chrono::steady_clock::now();

                my_mlp.ProcessNDisplayOneInputMtrx(kTestInput);

                end = std::chrono::steady_clock::now();
                elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                std::cout << "\nElapsed time Process Neurons by MatrixForm: " << elapsed_time << " miliseconds\n\n";
        // !Test of matrix form Processing      

                // Process all data in base
                std::vector<mlp::TensorT> test_inputs{};
                for (const file::tables::TableStudentsMark::RowType& row : database.data_rows) {
                    test_inputs.push_back(mlp::TensorT{ static_cast<double>(std::get<0>(row)), static_cast<double>(std::get<1>(row)) });
                }
                my_mlp.ProcessNDisplayAllInput(test_inputs);
                std::cout << "Process all data in base\n\n";

#endif // NODEFORM

#ifdef MATRIXFORM_STUDENTS_MARK
                start = std::chrono::steady_clock::now();

                for (double input_ptr : letters) {
                    mlp::TensorT target{ mlp::kTargetMax, mlp::kTargetMin, mlp::kTargetMin };
                    mlp::TensorT input_vec{ mlp::TensorT{ input_ptr } };
                    my_ann.LearnMatrixMLP(max_epoch_index, input_vec, target);
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

#ifdef LOADCUSTOM_STUDENTS_MARK
                mlp::MultiLayerPerceptron custom_mlp{};
                custom_mlp.LoadMLP("zero_classifier");
                mlp::ProcessNDisplay(TensorT{ 10000, 2000, 6000, -49999 }, custom_mlp);
#endif // LOADCUSTOM

#ifdef SAVENETWORK_STUDENTS_MARK
                constexpr char net_name[]{ "zero_classifier" };
                my_mlp.SaveMLP(net_name);
                mlp::MultiLayerPerceptron test_load{};
                test_load.LoadMLP(net_name);
#endif // SAVENETWORK


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

#endif // !STUDENTS_MARK_H