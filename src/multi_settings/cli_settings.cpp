#include "cli_settings.h"

namespace settings
{

// trim from start (in place)
static inline void ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) { return !std::isspace(ch); }));
}

// trim from end (in place)
static inline void rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) { return !std::isspace(ch); }).base(),
            s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s)
{
    ltrim(s);
    rtrim(s);
}

CLISettings::CLISettings(const std::vector<std::string> &settings)
{
    for (const auto &s : settings)
    {
        size_t pos = s.find("=");
        if (pos == std::string::npos)
            throw BadValue("Input setting of value '" + s +
                           "' could not be parsed, expected format KEY=VALUE");

        std::string key = s.substr(0, pos);
        trim(key);
        std::string val = s.substr(pos + 1);
        trim(val);

        if (key.empty() || val.empty())
            throw BadValue("Input setting of value '" + s +
                           "' could not be parsed, expected format KEY=VALUE");

        try
        {
            int v = std::stoi(val, &pos);
            if (pos == val.length())
            {
                mSettings.emplace(key, Value(std::make_shared<IntValue>(IntValue(key, v))));
                continue;
            }
        }
        catch (const std::invalid_argument &e)
        {
        }
        catch (const std::out_of_range &e)
        {
        }

        try
        {
            double v = std::stod(val, &pos);
            if (pos == val.length())
            {
                mSettings.emplace(key, Value(std::make_shared<DoubleValue>(DoubleValue(key, v))));
                continue;
            }
        }
        catch (const std::invalid_argument &e)
        {
        }
        catch (const std::out_of_range &e)
        {
        }

        mSettings.emplace(key, Value(std::make_shared<StringValue>(StringValue(key, val))));
    }
}

Value CLISettings::operator[](const std::string &key) const
{
    try
    {
        return mSettings.at(key);
    }
    catch (const std::out_of_range &e)
    {
        throw MissingValue("Value with name '" + key + "' does not exists");
    }
}

} // namespace settings