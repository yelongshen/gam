#pragma once

#if __has_include(<span>) && __cplusplus >= 202002L
#include <span>
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