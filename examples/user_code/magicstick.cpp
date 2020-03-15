// ----------------------------- OpenPose C++ API Tutorial - Example 1 - Body from image -----------------------------
// It reads an image, process it, and displays it with the pose keypoints.

// Third-party dependencies
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
// Command-line user interface
#define OPENPOSE_FLAGS_DISABLE_POSE
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

#include <cmath>

using namespace std;

#define STICK_RELATIVE_LENGTH 2
#define DURATION 90

queue<long> stick_point;


// Custom OpenPose flags
// Producer
DEFINE_string(image_path, "/home/lili/0.jpg",
    "Process an image. Read all standard formats (jpg, png, bmp, etc.).");
// DEFINE_string(image_path, "examples/media/COCO_val2014_000000000192.jpg",
//     "Process an image. Read all standard formats (jpg, png, bmp, etc.).");
// Display
DEFINE_bool(no_display,                 false,
    "Enable to disable the visual display.");

// This worker will just read and return all the jpg files in a directory
void display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    try
    {
        // User's displaying/saving/other processing here
            // datum.cvOutputData: rendered frame with pose or heatmaps
            // datum.poseKeypoints: Array<float> with the estimated pose
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            // Display image
            const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);
            long current_stick_end = stick_point.front();
            stick_point.pop();
            cv::Point stick_end(current_stick_end >> 16, current_stick_end && 0xFFFF);
            cv::circle(cvMat, stick_end, 5, cv::Scalar(0, 0, 255), -1);
            // cv::circle(cvMat, cv::Point(150,200), 100, cv::Scalar(0, 255, 0), -1);
            cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", cvMat);
            cv::waitKey(0);
        }
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    try
    {
        // Example: How to use the pose keypoints
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            // Alternative 1
            // op::opLog("Body keypoints: " + datumsPtr->at(0)->poseKeypoints.toString(), op::Priority::High);

            // // Alternative 2
            // op::opLog(datumsPtr->at(0)->poseKeypoints, op::Priority::High);

            // // Alternative 3
            // std::cout << datumsPtr->at(0)->poseKeypoints << std::endl;

            // Alternative 4 - Accesing each element of the keypoints
            op::opLog("\nKeypoints:", op::Priority::High);
            const auto& poseKeypoints = datumsPtr->at(0)->poseKeypoints;
            op::opLog("Person pose keypoints:", op::Priority::High);
            for (auto person = 0 ; person < poseKeypoints.getSize(0) ; person++)
            {
                op::opLog("Person " + std::to_string(person) + " (x, y, score):", op::Priority::High);
                for (auto bodyPart = 0 ; bodyPart < poseKeypoints.getSize(1) ; bodyPart++)
                {
                    std::string valueToPrint;
                    for (auto xyscore = 0 ; xyscore < poseKeypoints.getSize(2) ; xyscore++)
                        valueToPrint += std::to_string(   poseKeypoints[{person, bodyPart, xyscore}]   ) + " ";
                    op::opLog(valueToPrint, op::Priority::High);

                }

                //calculate the stick
                double RElbowx = poseKeypoints[{person, 3, 0}];
                double RElbowy = poseKeypoints[{person, 3, 1}];
                double RWristx = poseKeypoints[{person, 4, 0}];
                double RWristy = poseKeypoints[{person, 4, 1}];
                // double length_arm = sqrt(pow(RElbowx - RWristx, 2) + pow(RElbowy - RWristy, 2));
                long stick_end = ((int)(RWristx + STICK_RELATIVE_LENGTH * (RWristx - RElbowx))) << 16 + ((int)(RWristy + STICK_RELATIVE_LENGTH * (RWristy - RElbowy)));

                int a = (int)(RWristx + STICK_RELATIVE_LENGTH * (RWristx - RElbowx));
                int b = (int)(RWristy + STICK_RELATIVE_LENGTH * (RWristy - RElbowy));
                op::opLog("ais"+a+"bis"+b+"allis"+stick_end,op::Priority::High);
                // cv:Point stick_end(round(RWristx + STICK_RELATIVE_LENGTH * (RWristx - RElbowx)), rount(RWristy + STICK_RELATIVE_LENGTH * (RWristy - RElbowy));
                stick_point.push(stick_end);
            }
            op::opLog(" ", op::Priority::High);
        }
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

int tutorialApiCpp()
{
    try
    {
        op::opLog("Starting OpenPose demo...", op::Priority::High);
        const auto opTimer = op::getTimerInit();

        // Configuring OpenPose
        op::opLog("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();

        // Starting OpenPose
        op::opLog("Starting thread(s)...", op::Priority::High);
        opWrapper.start();

        // Process and display image
        const cv::Mat cvImageToProcess = cv::imread(FLAGS_image_path);
        const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cvImageToProcess);
        auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
        if (datumProcessed != nullptr)
        {
            printKeypoints(datumProcessed);
            if (!FLAGS_no_display)
                display(datumProcessed);
        }
        else
            op::opLog("Image could not be processed.", op::Priority::High);

        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

        // Return
        return 0;
    }
    catch (const std::exception&)
    {
        return -1;
    }
}

int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running tutorialApiCpp
    return tutorialApiCpp();
}
