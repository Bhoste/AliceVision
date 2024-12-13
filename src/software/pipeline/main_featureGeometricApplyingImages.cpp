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

#include <aliceVision/sfm/pipeline/regionsIO.hpp>
#include <aliceVision/sfm/pipeline/pairwiseMatchesIO.hpp>


#include <aliceVision/matching/matchesFiltering.hpp>
#include <aliceVision/matching/io.hpp>

#include <aliceVision/image/io.hpp>

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
    std::string imagesFolder;
    std::vector<std::string> predefinedPairList;
    std::string describerTypesName = feature::EImageDescriberType_enumToString(feature::EImageDescriberType::SIFT);
    int rangeStart = -1;
    int rangeSize = 0;
    std::vector<std::string> featuresFolders;
    std::vector<std::string> matchesFolders;

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
        ("describerTypes,d", 
            po::value<std::string>(&describerTypesName)->default_value(describerTypesName),
            feature::EImageDescriberType_informations().c_str())
        ("matchesFolders", 
            po::value<std::vector<std::string>>(&matchesFolders)->multitoken()->required(),
            "Path to folder(s) containing the input matches used for estimation.")
        ("imagesFolder", 
            po::value<std::string>(&imagesFolder)->required(),
            "Path to folder(s) containing the extracted features.");

    po::options_description optionalParams("Optional parameters");
    optionalParams.add_options()
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

    const std::vector<feature::EImageDescriberType> describerTypes = feature::EImageDescriberType_stringToEnums(describerTypesName);
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

    

    for (const auto & perDesc : pairwiseMatches)
    {
        Pair p = perDesc.first;
        IndexT referenceId = p.first;
        IndexT nextId = p.second;

        std::string path = imagesFolder + "/" + std::to_string(nextId) + "_" + std::to_string(referenceId)  + "_warp.exr";
        image::Image<image::RGBfColor> warp;
        image::readImage(path, warp, image::EImageColorSpace::NO_CONVERSION);

        const auto & referenceFeats = featuresPerView.getFeaturesPerDesc(referenceId);
        const auto & nextFeats = featuresPerView.getFeaturesPerDesc(nextId);

        for (const auto & matches : perDesc.second)
        {
            const auto & desc = matches.first;

            const feature::PointFeatures & referenceFeatures = referenceFeats.at(desc);
            const feature::PointFeatures & nextFeatures = nextFeats.at(desc);

            for (const auto & match : matches.second)
            {
                const feature::PointFeature & referenceFeature = referenceFeatures.at(match._i);
                const feature::PointFeature & nextFeature = nextFeatures.at(match._j);

                const Vec2 ref = referenceFeature.coords().cast<double>();
                const Vec2 next = nextFeature.coords().cast<double>();

                double ratioX = 5472.0 / 864;
                double ratioY = 3648.0 / 864;

                double x = ref.x() / ratioX;
                double y = ref.y() / ratioY;
                Vec2 cmp;

                if (y < 5 || x < 5) continue;
                if (y >= warp.height() - 5) continue;
                if (x >= warp.width() - 5) continue;

                double min = 1000000.0;
                image::RGBfColor pixmin;
                for (int i = -5; i <= 5; i++)
                {
                    for (int j = -5; j <= 5; j++)
                    {
                        image::RGBfColor pix;
                        pix = warp(std::floor(y) + i, std::floor(x) + j);                
                        cmp.x() = pix.g() * 5472.0;
                        cmp.y() = pix.r() * 3648.0;
                        double diff = (next - cmp).norm();
                        if (diff < min)
                        {
                            min = diff;
                            pixmin = pix;
                        }
                    }
                }
                


                if (min > 20.0)
                {
                    std::cout << ref.transpose() << std::endl;
                }
                   
            }
        }
    }

    /*std::stringstream ss;
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

    for (const auto & reconstructedPair : reconstructedPairs)
    {
        std::cout << reconstructedPair.reference << " " << reconstructedPair.next << std::endl;

        if (!(reconstructedPair.reference == 2002880582 && reconstructedPair.next == 2087034185))
        {
           //continue;
        }

        std::cout << reconstructedPair.model << std::endl;

        std::string path = imagesFolder + "/" + std::to_string(reconstructedPair.next) + "_" + std::to_string(reconstructedPair.reference)  + "_warp.exr";

        image::Image<image::RGBfColor> warp;
        image::readImage(path, warp, image::EImageColorSpace::NO_CONVERSION);

        double ratioX = 4032.0 / 864;
        double ratioY = 3024.0 / 864;

        robustEstimation::Mat3Model model(reconstructedPair.model);
        multiview::relativePose::FundamentalSymmetricEpipolarDistanceError errorObject;

        for (int i = 0; i < warp.height(); i++)
        {
            for (int j = 0; j < warp.width(); j++)
            {
                auto & pix = warp(i, j);
                if (pix.b() < 1e-12)
                {
                    pix.r() = 0.0f;
                    pix.g() = 0.0f;
                    pix.b() = 0.0f;
                    continue;
                }

                Vec2 ref;
                ref.x() = double(j) * ratioX;
                ref.y() = double(i) * ratioY;

                Vec2 next;
                next.x() = pix.g() * 4032.0;
                next.y() = pix.r() * 3024.0;


                double err = errorObject.error(model, ref, next);
                //std::cout << err << std::endl;
                if (sqrt(err)  < 24.0)//reconstructedPair.score)
                {
                    //count++;
                }
                else 
                {
                    pix.r() = 0.0f;
                    pix.g() = 0.0f;
                    pix.b() = 0.0f;
                }
            }
        }

        std::string outpath = matchesFolderOutput + "/" + std::to_string(reconstructedPair.next) + "_" + std::to_string(reconstructedPair.reference)  + "_warp.exr";
        image::writeImage(outpath, warp, image::ImageWriteOptions().toColorSpace(image::EImageColorSpace::NO_CONVERSION).storageDataType(image::EStorageDataType::Float));
    }*/

    return EXIT_SUCCESS;
}
