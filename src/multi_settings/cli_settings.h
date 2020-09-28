#pragma once

#include <map>

#include "settings.h"

namespace settings
{

class CLISettings : public Settings
{
public:
    CLISettings(const std::vector<std::string> &settings);
    ~CLISettings() = default;

    Value operator[](const std::string &key) const override;

private:
    std::map<std::string, Value> mSettings = {};
};

} // namespace settings