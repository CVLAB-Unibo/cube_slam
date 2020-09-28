#include "value.h"
#include <algorithm>

namespace settings
{

bool str2bool(const std::string &key, const std::string &s)
{
    std::string lowerCase = "";
    std::transform(
        s.begin(), s.end(), std::back_inserter(lowerCase),
        [](auto c) -> auto { return std::tolower(c); });

    if (lowerCase == "true")
        return true;
    else if (lowerCase == "false")
        return false;

    throw BadValue("Value named '" + key + "' not convertible to bool");
}

BaseValue::BaseValue(const std::string &key) : mKey(key) {}
const std::string &BaseValue::key() const { return mKey; }
BaseValue::operator bool() const
{
    throw BadValue("Value named '" + mKey + "' not convertible to bool");
}
BaseValue::operator int() const
{
    throw BadValue("Value named '" + mKey + "' not convertible to int");
}
BaseValue::operator float() const { return (double)*this; }
BaseValue::operator double() const
{
    throw BadValue("Value named '" + mKey + "' not convertible to double");
}
BaseValue::operator std::string() const
{
    throw BadValue("Value named '" + mKey + "' not convertible to string");
}

Value::Value(const std::shared_ptr<BaseValue> val) : BaseValue(""), mValue(val) {}
const std::string &Value::key() const { return mValue->key(); }
Value::operator bool() const { return (bool)*mValue; }
Value::operator int() const { return (int)*mValue; }
Value::operator float() const { return (float)*mValue; }
Value::operator double() const { return (double)*mValue; }
Value::operator std::string() const { return (std::string)*mValue; }

NodeValue::NodeValue(const std::string &key, const cv::FileNode &node) : BaseValue(key), mNode(node)
{
}

NodeValue::operator bool() const
{
    if (mNode.isString())
        return str2bool(mKey, *this);

    throw BadValue("Value named '" + mKey + "' not convertible to bool");
}
NodeValue::operator int() const
{
    if (mNode.isInt())
        return (int)mNode;

    throw BadValue("Value named '" + mKey + "' not convertible to int");
}
NodeValue::operator double() const
{
    if (mNode.isReal())
        return (double)mNode;
    else if (mNode.isInt())
        return (int)mNode;

    throw BadValue("Value named '" + mKey + "' not convertible to double");
}
NodeValue::operator std::string() const
{
    if (mNode.isString())
    {
        return std::string(((cv::String)mNode).c_str());
    }

    throw BadValue("Value named '" + mKey + "' not convertible to string");
}

StringValue::operator bool() const { return str2bool(mKey, mV); }

} // namespace settings