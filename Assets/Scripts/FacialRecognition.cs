using UnityEngine;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.IO;

using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

public class FacialRecognition : MonoBehaviour
{
    VideoCapture video;

    private Mat frame = new Mat();
    private CascadeClassifier classifier;
    private string path = "D:/Unity/Projects/facial-recognition/Assets/cascades/lbpcascade_frontalface_improved.xml";
    private Rectangle[] frontFaces;

    private int MIN_FACE_SIZE = 50;
    private int MAX_FACE_SIZE = 200;
       

    void Start()
    {
        video = new VideoCapture(0);
        classifier = new CascadeClassifier(fileName: path);

        if (video.IsOpened)
        {
            video.Grab();
            video.ImageGrabbed += new EventHandler(handleWebcamQueryFrame);

        }

    }

    void Update()
    {
        video.Grab();
    }

    void handleWebcamQueryFrame(object sender, EventArgs e)
    {
        if (video.IsOpened)
        {
            video.Retrieve(frame);
            if (frame.IsEmpty) return;

            Mat image = detectFace(frame).Mat;

            CvInvoke.Imshow("Video", image);
        }
    }

    void OnDestroy()
    {
        CvInvoke.DestroyAllWindows();
    }

    Image<Bgr, Byte> detectFace(Mat input)
    {
        Image<Bgr, Byte> image = input.ToImage<Bgr, Byte>();
        Bgr color = new Bgr(255, 0, 0);

        frontFaces = classifier.DetectMultiScale(image: image,
                                                 scaleFactor: 1.1,
                                                 minNeighbors: 5,
                                                 minSize: new Size(MIN_FACE_SIZE, MIN_FACE_SIZE),
                                                 maxSize: new Size(MAX_FACE_SIZE, MAX_FACE_SIZE));
        foreach( var face in frontFaces)
        {
            image.Draw(face, color, 3);
        }
        return image;
    }
}


