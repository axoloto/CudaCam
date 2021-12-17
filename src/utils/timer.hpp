#pragma once

#include <map>
#include <string>

struct timer
{
  double totalTime = 0.0;
  size_t nbCount = 0;
  float averageTime() const { return nbCount > 0 ? totalTime / nbCount : 0; }
};

class timerManager
{
public:
  static timerManager &Get()
  {
    static timerManager manager;
    return manager;
  }

  void createTimer(std::string name)
  {
    m_timers.insert(std::make_pair(name, timer()));
  }

  void addTime(std::string name, double time)
  {
    auto it = m_timers.find(name);
    if (it != m_timers.end())
    {
      it->second.totalTime += time;
      it->second.nbCount++;
    }
    else
    {
      LOG_ERROR("Timer {} unknown", name);
    }
  }

  double getAverageTime(std::string name) const
  {
    auto it = m_timers.find(name);
    if (it != m_timers.end() && it->second.nbCount > 0)
    {
      return it->second.totalTime / it->second.nbCount;
    }
    else
    {
      LOG_ERROR("Timer {} unknown", name);
      return 0.0;
    }
  }

  std::map<std::string, timer>::const_iterator beginTimerList() const { return m_timers.cbegin(); }
  std::map<std::string, timer>::const_iterator endTimerList() const { return m_timers.cend(); }

private:
  timerManager() = default;
  ~timerManager() = default;
  timerManager(const timerManager &) = delete;
  timerManager &operator=(const timerManager &) = delete;
  timerManager(timer &&) = delete;
  timerManager &operator=(const timer &&) = delete;

  std::map<std::string, timer> m_timers;
};