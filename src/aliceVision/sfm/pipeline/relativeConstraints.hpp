// This file is part of the AliceVision project.
// Copyright (c) 2023 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/types.hpp>
#include <aliceVision/numeric/numeric.hpp>
#include <aliceVision/dataio/json.hpp>
#include <aliceVision/geometry/lie.hpp>

namespace aliceVision {
namespace sfm {

struct ConstraintPair
{
    IndexT reference;
    IndexT next;
    Eigen::Matrix3d model;
    double score;
};

void tag_invoke(const boost::json::value_from_tag&, boost::json::value& jv, sfm::ConstraintPair const& input)
{
    jv = {{"reference", input.reference},
          {"next", input.next},
          {"model", boost::json::value_from(input.model)},
          {"score", boost::json::value_from(input.score)}};
}

ConstraintPair tag_invoke(boost::json::value_to_tag<ConstraintPair>, boost::json::value const& jv)
{
    const boost::json::object& obj = jv.as_object();

    ConstraintPair ret;

    ret.reference = boost::json::value_to<IndexT>(obj.at("reference"));
    ret.next = boost::json::value_to<IndexT>(obj.at("next"));
    ret.model = boost::json::value_to<Eigen::Matrix3d>(obj.at("model"));
    ret.score = boost::json::value_to<double>(obj.at("score"));

    return ret;
}

}  // namespace sfm
}  // namespace aliceVision