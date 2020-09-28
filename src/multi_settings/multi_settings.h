#pragma once

#include "settings.h"

namespace settings
{

class MultiSettings : public Settings
{
public:
    MultiSettings() = default;
    virtual ~MultiSettings() = default;

    template <class S> void add(const S &settings)
    {
        mSettings.push_back(std::make_shared<S>(settings));
    }
    Value operator[](const std::string &key) const override;

private:
    std::vector<std::shared_ptr<Settings>> mSettings = {};
};

} // namespace settings
