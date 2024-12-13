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
    std::string matchesFolderOutput;
    std::vector<std::string> featuresFolders;
    std::vector<std::string> matchesFolders;
    std::string geometricFilterTypeName =
      matchingImageCollection::EGeometricFilterType_enumToString(matchingImageCollection::EGeometricFilterType::FUNDAMENTAL_MATRIX);
    std::vector<std::string> predefinedPairList;
    std::string describerTypesName = feature::EImageDescriberType_enumToString(feature::EImageDescriberType::SIFT);
    int rangeStart = -1;
    int rangeSize = 0;
    int maxIteration = 50000;
    double geometricErrorMax = 0.0; 
    robustEstimation::ERobustEstimator geometricEstimator = robustEstimation::ERobustEstimator::ACRANSAC;
    int randomSeed = std::mt19937::default_seed;

    // clang-format off
    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()
        ("input,i", po::value<std::string>(&sfmDataFilename)->required(),
             "SfMData file.")
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
        ("geometricEstimator", 
            po::value<robustEstimation::ERobustEstimator>(&geometricEstimator)->default_value(geometricEstimator),
            "Geometric estimator:\n"
            "* acransac: A-Contrario Ransac\n"
            "* loransac: LO-Ransac (only available for fundamental matrix). Need to set '--geometricError'")
        ("describerTypes,d", 
            po::value<std::string>(&describerTypesName)->default_value(describerTypesName),
            feature::EImageDescriberType_informations().c_str())
        ("geometricFilterType,g", 
            po::value<std::string>(&geometricFilterTypeName)->default_value(geometricFilterTypeName))
        ("imagePairsList,l", 
            po::value<std::vector<std::string>>(&predefinedPairList)->multitoken(),
            "Path(s) to one or more files which contain the list of image pairs to match.")
        ("maxIteration", 
            po::value<int>(&maxIteration)->default_value(maxIteration),
            "Maximum number of iterations allowed in Ransac step.")
        ("geometricError", 
            po::value<double>(&geometricErrorMax)->default_value(geometricErrorMax),
            "Maximum error (in pixels) allowed for features matching during geometric verification. "
            "If set to 0, it lets the ACRansac select an optimal value.")
        ("randomSeed", 
            po::value<int>(&randomSeed)->default_value(randomSeed),
            "This seed value will generate a sequence using a linear random generator. Set -1 to use a random seed.")
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


    // Build the list of image pairs to process
    // from matching mode compute the pair list that have to be matched
    PairSet pairs;
    std::set<IndexT> filter;

    // We assume that there is only one pair for (I,J) and (J,I)
    if (predefinedPairList.empty())
    {
        pairs = exhaustivePairs(sfmData.getViews(), rangeStart, rangeSize);
    }
    else
    {
        for (const std::string& imagePairsFile : predefinedPairList)
        {
            ALICEVISION_LOG_INFO("Load pair list from file: " << imagePairsFile);
            if (!matchingImageCollection::loadPairsFromFile(imagePairsFile, pairs, rangeStart, rangeSize))
            {
                return EXIT_FAILURE;
            }
        }
    }

    for (const auto & pair : pairs)
    {
        Pair symmetry;
        symmetry.first = pair.second;
        symmetry.second = pair.first;
        pairs.insert(symmetry);
    }
    
    if (pairs.empty())
    {
        ALICEVISION_LOG_INFO("No image pair to match.");
        // if we only compute a selection of matches, we may have no match.
        return rangeSize ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const std::vector<feature::EImageDescriberType> describerTypes = 
        feature::EImageDescriberType_stringToEnums(describerTypesName);
    const matchingImageCollection::EGeometricFilterType geometricFilterType =
      matchingImageCollection::EGeometricFilterType_stringToEnum(geometricFilterTypeName);
    std::mt19937 randomNumberGenerator(randomSeed == -1 ? std::random_device()() : randomSeed);

    const double defaultLoRansacMatchingError = 20.0;
    if (!adjustRobustEstimatorThreshold(geometricEstimator, geometricErrorMax, defaultLoRansacMatchingError))
    {
        return EXIT_FAILURE;
    }
    
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
    ss << matchesFolderOutput << "/pairs_" << rangeStart << ".json";
    std::ofstream of(ss.str());

    std::vector<sfm::ConstraintPair> reconstructedPairs;
    for (auto pair : pairs)
    {
        IndexT referenceId = pair.first;
        IndexT nextId = pair.second;

        bool inverse = false;
        if (pairwiseMatches.count(pair) == 0)
        {
            std::swap(pair.first, pair.second);
            inverse = true;
        }

        if (pairwiseMatches.count(pair) == 0)
        {
            continue;
        }

        const auto & perDesc = pairwiseMatches.at(pair);

        std::cout << referenceId << " " << nextId << std::endl;
                
        const auto & referenceFeats = featuresPerView.getFeaturesPerDesc(referenceId);
        const auto & nextFeats = featuresPerView.getFeaturesPerDesc(nextId);

        size_t refWidth = sfmData.getView(referenceId).getImageInfo()->getWidth();
        size_t refHeight = sfmData.getView(referenceId).getImageInfo()->getHeight();
        size_t nextWidth = sfmData.getView(nextId).getImageInfo()->getWidth();
        size_t nextHeight = sfmData.getView(nextId).getImageInfo()->getHeight();

        size_t count = 0;
        for (const auto & [desc, matches] : perDesc)
        {
            count += matches.size();
        }

        Mat x1(2, count);
        Mat x2(2, count);

        

        size_t pos = 0;
        for (const auto & [desc, matches] : perDesc)
        {
            const feature::PointFeatures & referenceFeatures = referenceFeats.at(desc);
            const feature::PointFeatures & nextFeatures = nextFeats.at(desc);

            for (const matching::IndMatch& match : matches)
            {
                const feature::PointFeature & referenceFeature = referenceFeatures.at(match._i);
                const feature::PointFeature & nextFeature = nextFeatures.at(match._j);

                x1.col(pos) = referenceFeature.coords().cast<double>();
                x2.col(pos) = nextFeature.coords().cast<double>();

                pos++;
            }
        }

        if (inverse)
        {
            std::swap(x1, x2);
        }

        using SolverT = multiview::relativePose::Fundamental7PSolver;
        using ModelT = robustEstimation::Mat3Model;

        // define the AContrario adapted Fundamental matrix solver
        using KernelT = multiview::RelativePoseKernel<SolverT,
                                                      multiview::relativePose::FundamentalSymmetricEpipolarDistanceError,
                                                      multiview::UnnormalizerT,
                                                      ModelT>;

        const KernelT kernel(x1, refWidth, refHeight, x2, nextWidth, nextHeight, true);

        // robustly estimate the Fundamental matrix with A Contrario ransac
        const double upperBoundPrecision = geometricErrorMax;

        ModelT model;
        std::vector<std::size_t> out_inliers;
        const std::pair<double, double> ACRansacOut = robustEstimation::ACRANSAC(kernel, randomNumberGenerator, out_inliers, maxIteration, &model, geometricErrorMax);
        Eigen::Matrix3d m_F = model.getMatrix();

        sfm::ConstraintPair outputPair;
        outputPair.reference = referenceId;
        outputPair.next = nextId;
        outputPair.model = m_F;
        outputPair.score = ACRansacOut.first;

        std::cout << out_inliers.size() << std::endl;

        count = 0;
        multiview::relativePose::FundamentalEpipolarDistanceError err;
        for (int i = 0; i < pos; i++)
        {
            if (sqrt(err.error(model, x1.col(i), x2.col(i))) < ACRansacOut.first)
            {
                count++;
            }
        }
        
        if (count < 1000)
        {
            continue;
        }

        reconstructedPairs.push_back(outputPair);
    }


    boost::json::value jv = boost::json::value_from(reconstructedPairs);
    of << boost::json::serialize(jv);
    of.close();

    return EXIT_SUCCESS;
}
