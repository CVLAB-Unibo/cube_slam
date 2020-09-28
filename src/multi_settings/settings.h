#pragma once

#include <opencv2/core/persistence.hpp>

#include "value.h"

namespace settings
{

class Settings
{
public:
    Settings() = default;
    virtual ~Settings() = default;

    virtual Value operator[](const std::string &key) const = 0;
};

} // namespace settings
