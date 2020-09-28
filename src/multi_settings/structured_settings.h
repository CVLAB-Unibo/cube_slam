#pragma once

#include "settings.h"

namespace settings
{

class StructuredSettings : public Settings
{
public:
    StructuredSettings(const std::string &filename);
    virtual ~StructuredSettings() = default;

    Value operator[](const std::string &key) const override;

private:
    cv::FileStorage mStorage;
};

} // namespace settings