#ifndef ERRORS_MLP_H

#include <string>
#include <exception>
#include <stdexcept>

namespace mlp {

    constexpr char kErrorCodeMsg[]{ " Error code: " };
    constexpr unsigned int kErrorCodeRuntimeMLP    { 1 };
    constexpr unsigned int kErrorCodeLogicMLP      { 2 };
    constexpr unsigned int kErrorCodeSaveMLP       { 3 };
    constexpr unsigned int kErrorCodeLoadMLP       { 4 };

    struct ErrorRuntimeMLP : public std::runtime_error {
        const unsigned int code{ kErrorCodeRuntimeMLP };
        ErrorRuntimeMLP(const std::string& msg)
            : code{ kErrorCodeRuntimeMLP },
            std::runtime_error(msg + kErrorCodeMsg + std::to_string(code)) {
        }
    };
    struct ErrorLogicMLP : public std::logic_error {
        const unsigned int code{ kErrorCodeLogicMLP };
        ErrorLogicMLP(const std::string& msg) : std::logic_error(msg + kErrorCodeMsg + std::to_string(code)) {}
    };
    struct ErrorSaveMLP : public std::runtime_error {
        const unsigned int code{ kErrorCodeSaveMLP };
        ErrorSaveMLP(const std::string& msg) : std::runtime_error(msg + kErrorCodeMsg + std::to_string(code)) {}
    };
    struct ErrorLoadMLP : public std::runtime_error {
        const unsigned int code{ kErrorCodeLoadMLP };
        ErrorLoadMLP(const std::string& msg) : std::runtime_error(msg + kErrorCodeMsg + std::to_string(code)) {}
    };

}

#define ERRORS_MLP_H
#endif // !ERRORS_MLP_H