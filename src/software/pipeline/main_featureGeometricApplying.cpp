// This file is part of the AliceVision project.
// Copyright (c) 2024 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/system/main.hpp>
#include <aliceVision/system/Timer.hpp>
#include <aliceVision/cmdline/cmdline.hpp>

#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>

#include <aliceVision/matchingImageCollection/GeometricFilterType.hpp>
#include <aliceVision/matchingImageCollection/pairBuilder.hpp>
#include <aliceVision/matchingImageCollection/ImagePairListIO.hpp>

#include <aliceVision/robustEstimation/ACRansac.hpp>
#include <aliceVision/robustEstimation/estimators.hpp>
#include <aliceVision/multiview/relativePose/Fundamental7PSolver.hpp>
#include <aliceVision/multiview/relativePose/FundamentalError.hpp>
#include <aliceVision/multiview/RelativePoseKernel.hpp>
#include <aliceVision/multiview/Unnormalizer.hpp>


#include <aliceVision/matching/matchesFiltering.hpp>
#include <aliceVision/matching/io.hpp>

#include <aliceVision/feature/RegionsPerView.hpp>
#include <aliceVision/sfm/pipeline/regionsIO.hpp>
#include <aliceVision/sfm/pipeline/pairwiseMatchesIO.hpp>

#include <aliceVision/matching/IndMatch.hpp>
#include <aliceVision/sfm/pipeline/relativeConstraints.hpp>

#include <boost/program_options.hpp>

// These constants define the current software version.
// They must be updated when the command line is changed.
#define ALICEVISION_SOFTWARE_VERSION_MAJOR 1
#define ALICEVISION_SOFTWARE_VERSION_MINOR 0

namespace po = boost::program_options;

using namespace aliceVision;

int aliceVision_main(int argc, char** argv)
{
    // command-line parameters
    std::string sfmDataFilename;
    std::string estimationFolder;
    std::string matchesFolderOutput;
    std::vector<std::string> featuresFolders;
    std::vector<std::string> matchesFolders;
    std::vector<std::string> predefinedPairList;
    std::string describerTypesName = feature::EImageDescriberType_enumToString(feature::EImageDescriberType::SIFT);
    int rangeStart = -1;
    int rangeSize = 0;

    // clang-format off
    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()
        ("input,i", po::value<std::string>(&sfmDataFilename)->required(),
             "SfMData file.")
        ("estimationFolder", po::value<std::string>(&estimationFolder)->required(),
             "Estimation folder.")
        ("output,o", po::value<std::string>(&matchesFolderOutput)->required(),
            "Path to a folder in which computed matches will be stored.")

        ("featuresFolders", 
            po::value<std::vector<std::string>>(&featuresFolders)->multitoken()->required(),
            "Path to folder(s) containing the extracted features.")
        ("matchesFolders", 
            po::value<std::vector<std::string>>(&matchesFolders)->multitoken()->required(),
            "Path to folder(s) containing the input matches used for estimation.");

    po::options_description optionalParams("Optional parameters");
    optionalParams.add_options()
        ("describerTypes,d", 
            po::value<std::string>(&describerTypesName)->default_value(describerTypesName),
            feature::EImageDescriberType_informations().c_str())
        ("imagePairsList,l", 
            po::value<std::vector<std::string>>(&predefinedPairList)->multitoken(),
            "Path(s) to one or more files which contain the list of image pairs to match.")
        ("rangeStart", 
            po::value<int>(&rangeStart)->default_value(rangeStart),
            "Range image index start.")
        ("rangeSize", 
            po::value<int>(&rangeSize)->default_value(rangeSize),
            "Range size.");

    // clang-format on

    CmdLine cmdline("This program filters input matches according to a geometric model:\n"
                    "AliceVision featureGeometricFiltering");

    cmdline.add(requiredParams);
    cmdline.add(optionalParams);
    if (!cmdline.execute(argc, argv))
    {
        return EXIT_FAILURE;
    }

    sfmData::SfMData sfmData;
    if (!sfmDataIO::load(sfmData, sfmDataFilename, 
            sfmDataIO::ESfMData(
                sfmDataIO::VIEWS | sfmDataIO::INTRINSICS | sfmDataIO::EXTRINSICS
            )
        ))
    {
        ALICEVISION_LOG_ERROR("The input SfMData file '" << sfmDataFilename << "' cannot be read.");
        return EXIT_FAILURE;
    }


    const std::vector<feature::EImageDescriberType> describerTypes = 
        feature::EImageDescriberType_stringToEnums(describerTypesName);

    
    // features reading
    feature::FeaturesPerView featuresPerView;
    ALICEVISION_LOG_INFO("Load features");
    if (!sfm::loadFeaturesPerView(featuresPerView, sfmData, featuresFolders, describerTypes))
    {
        ALICEVISION_LOG_ERROR("Invalid features.");
        return EXIT_FAILURE;
    }

    // matches reading
    matching::PairwiseMatches pairwiseMatches;
    ALICEVISION_LOG_INFO("Load features matches");
    if (!sfm::loadPairwiseMatches(
          pairwiseMatches, sfmData, matchesFolders, describerTypes, 0, 0, true))
    {
        ALICEVISION_LOG_ERROR("Unable to load matches.");
        return EXIT_FAILURE;
    }

    std::stringstream ss;
    ss << estimationFolder << "/pairs_" << rangeStart << ".json";
    std::ifstream inputfile(ss.str());
    if (!inputfile.is_open())
    {
        ALICEVISION_LOG_INFO("No input file found for estimation");
        return EXIT_SUCCESS;
    }

    std::stringstream buffer;
    buffer << inputfile.rdbuf();
    boost::json::value jv = boost::json::parse(buffer.str());

    std::vector<sfm::ConstraintPair> reconstructedPairs = boost::json::value_to<std::vector<sfm::ConstraintPair>>(jv);

    multiview::relativePose::FundamentalEpipolarDistanceError errorObject;

    matching::PairwiseMatches pairwiseMatchesOutput;
    for (const auto & constraint : reconstructedPairs)
    {
        Pair pair;
        pair.first = constraint.reference;
        pair.second = constraint.next;

        const auto & perDesc = pairwiseMatches.at(pair);
        auto & perDescOutput = pairwiseMatchesOutput[pair];
        
        robustEstimation::Mat3Model model(constraint.model);
                
        const auto & referenceFeats = featuresPerView.getFeaturesPerDesc(pair.first);
        const auto & nextFeats = featuresPerView.getFeaturesPerDesc(pair.second);

        int count = 0;
        for (const auto & [desc, matches] : perDesc)
        {
            const feature::PointFeatures & referenceFeatures = referenceFeats.at(desc);
            const feature::PointFeatures & nextFeatures = nextFeats.at(desc);

            auto & matchesOutput = perDescOutput[desc];

            for (const matching::IndMatch& match : matches)
            {
                const feature::PointFeature & referenceFeature = referenceFeatures.at(match._i);
                const feature::PointFeature & nextFeature = nextFeatures.at(match._j);

                const Vec2 & ref = referenceFeature.coords().cast<double>();
                const Vec2 & next = nextFeature.coords().cast<double>();

                double err = errorObject.error(model, ref, next);
                if (err  < constraint.score)
                {
                    count++;
                }

                matchesOutput.push_back(match);
            }
        }      
    }

    inputfile.close();

    // when a range is specified, generate a file prefix to reflect the current iteration (rangeStart/rangeSize)
    // => with matchFilePerImage: avoids overwriting files if a view is present in several iterations
    // => without matchFilePerImage: avoids overwriting the unique resulting file
    const std::string filePrefix = rangeSize > 0 ? std::to_string(rangeStart / rangeSize) + "." : "";

    const std::string fileExtension = "txt";
    matching::Save(pairwiseMatchesOutput, matchesFolderOutput, fileExtension, false, filePrefix);

    return EXIT_SUCCESS;
}
