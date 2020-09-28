#include "structured_settings.h"

namespace settings
{

StructuredSettings::StructuredSettings(const std::string &filename)
    : Settings(), mStorage(cv::FileStorage(filename.c_str(), cv::FileStorage::READ))
{
    if (!mStorage.isOpened())
    {
        throw std::runtime_error("Settings file at '" + filename + "' cannot be opened");
    }
}

Value StructuredSettings::operator[](const std::string &key) const
{
    auto node = mStorage[key.c_str()];
    if (node.empty() || node.isNone())
        throw MissingValue("Value with name '" + key + "' does not exists");

    return Value(std::make_shared<NodeValue>(NodeValue(key, node)));
}

} // namespace settings