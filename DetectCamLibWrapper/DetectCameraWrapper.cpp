#include "DetectCameraWrapper.h"

System::Collections::Generic<System::String>^ ProcessFrameWrapper(
    OpenCvSharp::Mat^ mat)
{
    var names = gcnew System::Collections::Generic<System::String>();
    auto matNativePtr =
        reinterpret_cast<cv::Mat*>(marshal_as<void*>(mat->CvPtr));
    auto namesNative = ProcessFrame(*matNativePtr);
    for (const auto& nameNative : namesNative)
    {
        names->Add(marshal_as<System::String^>(nameNative));
    }
    return names;
}
