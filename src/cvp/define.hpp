#pragma once

#include <map>
#include <string>

namespace cvp
{
// List of Canny Edges
enum CannyStage
{
  MONO = 0,
  GAUSSIAN = 1,
  GRADIENT = 2,
  NMS = 3,
  THRESH = 4,
  HYSTER = 5
};

struct CompareCannyStage
{
  bool operator()(const CannyStage &stageA, const CannyStage &stageB) const
  {
    return (int)stageA < (int)stageB;
  }
};

static const std::map<CannyStage, std::string, CompareCannyStage> CANNY_STAGES{
  { CannyStage::MONO, "1/6 Mono Conversion" },
  { CannyStage::GAUSSIAN, "2/6 Gaussian Noise Removal" },
  { CannyStage::GRADIENT, "3/6 Gradient Computation" },
  { CannyStage::NMS, "4/6 Non Maximum Suppression" },
  { CannyStage::THRESH, "5/6 Double Threshold" },
  { CannyStage::HYSTER, "6/6 Hysteresis" }
};

}// namespace cvp