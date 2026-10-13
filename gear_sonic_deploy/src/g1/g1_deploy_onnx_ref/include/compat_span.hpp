#pragma once

// Some compilers (e.g. g++-10 in -std=c++2a mode) report a pre-C++20
// __cplusplus value even though libstdc++'s <span> is fully implemented and
// usable. Relying on __cplusplus here caused this polyfill to activate
// alongside the real std::span, producing "redeclared with 1 template
// parameter" errors. Prefer the __cpp_lib_span feature-test macro, which
// libstdc++ defines only when a real, usable std::span is present.
#if __has_include(<span>)
#include <span>
#endif

#if defined(__cpp_lib_span)
// Real std::span is available; nothing further to do.
#else

#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace std {

template <typename T>
class span {
public:
  using element_type = T;
  using value_type = typename std::remove_cv<T>::type;
  using size_type = std::size_t;
  using iterator = T*;

  span() : data_(nullptr), size_(0) {}
  span(T* data, size_type size) : data_(data), size_(size) {}

  template <typename U, typename std::enable_if<std::is_const<T>::value && std::is_same<typename std::remove_const<T>::type, U>::value, int>::type = 0>
  span(const span<U>& other) : data_(other.data()), size_(other.size()) {}

  template <std::size_t N>
  span(std::array<value_type, N>& arr) : data_(arr.data()), size_(N) {}

  template <std::size_t N, typename U = T, typename std::enable_if<std::is_const<U>::value, int>::type = 0>
  span(const std::array<value_type, N>& arr) : data_(arr.data()), size_(N) {}

  span(std::vector<value_type>& vec) : data_(vec.data()), size_(vec.size()) {}

  template <typename U = T, typename std::enable_if<std::is_const<U>::value, int>::type = 0>
  span(const std::vector<value_type>& vec) : data_(vec.data()), size_(vec.size()) {}

  iterator begin() const { return data_; }
  iterator end() const { return data_ + size_; }
  T* data() const { return data_; }
  size_type size() const { return size_; }
  bool empty() const { return size_ == 0; }

private:
  T* data_;
  size_type size_;
};

template <typename T, std::size_t N>
span(std::array<T, N>&) -> span<T>;

template <typename T, std::size_t N>
span(const std::array<T, N>&) -> span<const T>;

template <typename T>
span(std::vector<T>&) -> span<T>;

template <typename T>
span(const std::vector<T>&) -> span<const T>;

}  // namespace std

#endif