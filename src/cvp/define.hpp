#pragma once

#include <map>
#include <string>

namespace cvp
{
// List of Canny Edges
enum CannyStage
{
  MONO = 0,
  NOISE = 1,
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
  { CannyStage::MONO, "Mono Conversion" },
  { CannyStage::NOISE, "Gaussian Noise Removal" },
  { CannyStage::GRADIENT, "Gradient Computation" },
  { CannyStage::NMS, "Non Maximum Suppression" },
  { CannyStage::THRESH, "Double Threshold" },
};
}// namespace cvp