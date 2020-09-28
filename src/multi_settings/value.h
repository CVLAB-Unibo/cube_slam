#pragma once

#include <memory>
#include <opencv2/core/persistence.hpp>
#include <stdexcept>

namespace settings
{

class BadValue : public std::runtime_error
{
public:
    BadValue(const std::string &msg) : std::runtime_error(msg) {}
    ~BadValue() = default;
};

class MissingValue : public std::runtime_error
{
public:
    MissingValue(const std::string &msg) : std::runtime_error(msg) {}
    ~MissingValue() = default;
};

class BaseValue
{
public:
    BaseValue(const std::string &key);
    ~BaseValue() = default;

    virtual operator bool() const;
    virtual operator int() const;
    virtual operator float() const;
    virtual operator double() const;
    virtual operator std::string() const;

    virtual const std::string &key() const;

protected:
    std::string mKey;
};

class Value : public BaseValue
{
public:
    Value(const std::shared_ptr<BaseValue> val);
    ~Value() = default;

    operator bool() const override;
    operator int() const override;
    operator float() const override;
    operator double() const override;
    operator std::string() const override;

    const std::string &key() const override;

private:
    std::shared_ptr<BaseValue> mValue;
};

class NodeValue : public BaseValue
{
public:
    NodeValue(const std::string &key, const cv::FileNode &node);
    ~NodeValue() = default;

    operator bool() const override;
    operator int() const override;
    operator double() const override;
    operator std::string() const override;

private:
    cv::FileNode mNode;
};

class IntValue : public BaseValue
{
public:
    IntValue(const std::string &key, const int v) : BaseValue(key), mV(v) {}
    ~IntValue() = default;

    operator int() const override { return mV; }
    operator double() const override { return mV; }

private:
    int mV;
};

class DoubleValue : public BaseValue
{
public:
    DoubleValue(const std::string &key, const double v) : BaseValue(key), mV(v) {}
    ~DoubleValue() = default;

    operator int() const override { return mV; }
    operator double() const override { return mV; }

private:
    double mV;
};

class StringValue : public BaseValue
{
public:
    StringValue(const std::string &key, const std::string &v) : BaseValue(key), mV(v) {}
    ~StringValue() = default;

    operator bool() const override;
    operator std::string() const override { return mV; }

private:
    std::string mV;
};

} // namespace settings