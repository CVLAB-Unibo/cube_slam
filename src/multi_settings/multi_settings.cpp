#include "multi_settings.h"

namespace settings
{

Value MultiSettings::operator[](const std::string &key) const
{
    for (auto i = mSettings.rbegin(); i != mSettings.rend(); ++i)
    {
        const auto &settings = *i;
        try
        {
            return (*settings)[key];
        }
        catch (const MissingValue &e)
        {
            continue;
        }
    }

    throw MissingValue("Value with name '" + key + "' does not exists in any given settings");
}

} // namespace settings