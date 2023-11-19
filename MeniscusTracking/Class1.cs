using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using MeniscusTracking;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Emgu.Util;
using static System.Net.Mime.MediaTypeNames;


namespace MeniscusTracking
{
        public class Class1
        {
            public struct CCStatsOp
            {
                public Rectangle Rectangle;
                public int Area;
            }

            private Mat myErode(Mat src, int val)
            {
                int erosion_size = val;
                var dest = new Mat();
                CvInvoke.Erode(src, dest, null, new Point(-1, -1), val, BorderType.Default, CvInvoke.MorphologyDefaultBorderValue);
                return dest;
            }


            public int MeniscusTop(string inFileName, int frameStart, int frameEnd)
            {
                int frameno;
                int minval = int.MaxValue;

                var capture = new VideoCapture(inFileName);
                Mat frame0 = new Mat();
                BackgroundSubtractorMOG2 backSub = new BackgroundSubtractorMOG2();
                int totFrames = (int)capture.Get(CapProp.FrameCount);
                if (frameEnd <= frameStart)
                {
                    frameEnd = totFrames;
                }

                capture.Set(CapProp.PosFrames, frameStart);
                if (!capture.IsOpened)
                {
                    System.Console.WriteLine("Unable to open: " + inFileName);
                    System.Environment.Exit(0);
                }
                while (true)
                {
                    capture.Read(frame0);
                    if (frame0.IsEmpty)
                        break;
                    frameno = (int)capture.Get(CapProp.PosFrames);
                    if (frameno > frameEnd)
                        break;
                    Mat fgMask0 = new Mat();
                    backSub.Apply(frame0, fgMask0);
                    Rectangle rect = new Rectangle(10, 2, 100, 20);
                    CvInvoke.Rectangle(frame0, rect, new MCvScalar(255, 255, 255));
                    string label = frameno.ToString();
                    CvInvoke.PutText(frame0, label, new Point(15, 15),
                                FontFace.HersheySimplex, 0.5, new MCvScalar(0, 0, 0));

                    CvInvoke.Imshow("Frame", frame0);
                    var frame1 = myErode(fgMask0, 2);
                    CvInvoke.Imshow("FG Mask", frame1);
                    CvInvoke.WaitKey(30);
                    var ret = CvInvoke.BoundingRectangle(frame1);
                    if (ret.Top != 0)
                    {
                        minval = Math.Min(minval, ret.Top);
                        System.Console.WriteLine(frameno.ToString("G") + " " + ret.Top.ToString("G"));
                    }
                }
                return minval;
            }

            public double MeniscusFrom2Img()
            {
                /*This routine expects 2 frames in Directory C:\ProgramData\LabScript\Videos\FluidDetection,
                 * named Before.bmp and After.bmp. It compares the 2 images and isolates regions with more than 10
                 * gray levels of difference, cleans up noise and reports the difference between top and bottom
                 * of the bounding box of the largest area connected component. It moves the 2 original image files to 
                 * folder Archive within the above folder, appending time/date info to the image names
                 * TODO: calibration, Region-of-interest isolation, sanity check image display removal
                 * Throws exceptions if reading files or renaming fails returnin Nan in those cases
                */

                double delta = int.MaxValue;
                string InFolder = "C:\\ProgramData\\LabScript\\Videos\\FluidDetection";
                string Image1Name = "Before"; //expected image name
                string Image2Name = "After";  //expected image name
                Image<Rgb, byte> img1; //color image
                Image<Rgb, byte> img2; //color image
                Image<Gray, byte> gray1; //gray scale image
                Image<Gray, byte> gray2; //gray scale image
                int threshold = 10; //threshold value for image binarization


                try
                {
                    string im1 = InFolder + "\\" + Image1Name + ".bmp";
                    string im2 = InFolder + "\\" + Image2Name + ".bmp";
                    // read-in images and convert to grayscale
                    img1 = new Image<Rgb, Byte>(im1);
                    gray1 = new Image<Gray, byte>(img1.Rows, img1.Cols);
                    CvInvoke.CvtColor(img1, gray1, Emgu.CV.CvEnum.ColorConversion.Rgb2Gray);
                    img2 = new Image<Rgb, Byte>(im2);// path can be absolute or relative.
                    gray2 = new Image<Gray, byte>(img2.Rows, img2.Cols);
                    CvInvoke.CvtColor(img2, gray2, Emgu.CV.CvEnum.ColorConversion.Rgb2Gray);
                    //Subtract 2 images (absolute-value difference)
                    gray1 = gray1.AbsDiff(gray2);
                    //Threshold result
                    gray2 = gray1.ThresholdBinary(new Gray(threshold), new Gray(255)).Erode(5).Dilate(5);

                }
                catch (Exception ex)
                {
                    System.Console.WriteLine("Error in reading file:" + ex.Message);
                    return Double.NaN;
                }


                Mat imgLabel = new Mat();
                Mat stats = new Mat();
                Mat centroids = new Mat();
                //Run connected components analysis
                int nLabel = CvInvoke.ConnectedComponentsWithStats(gray2, imgLabel, stats, centroids);
                CCStatsOp[] statsOp = new CCStatsOp[stats.Rows];
                stats.CopyTo(statsOp);
                // Find the largest-area non background component.
                // Note: range() starts from 1 since 0 is the background label.
                int maxval = -1;
                int maxLabel = -1;
                Rectangle rect1 = new Rectangle(0, 0, 0, 0);
                for (int i = 1; i < nLabel; i++)
                {
                    int temp = statsOp[i].Area;
                    if (temp > maxval)
                    {
                        maxval = temp;
                        maxLabel = i;
                        rect1 = statsOp[i].Rectangle;
                    }
                }
                //Display result for sanity check
                gray2.Draw(rect1, new Gray(64));
                CvInvoke.Imshow("Rect", gray2);
                CvInvoke.WaitKey(3000);
                //Move files to Archive folder
                if (rect1.Top != 0)
                {
                    delta = rect1.Bottom - rect1.Top;
                    System.Console.WriteLine(rect1.Top.ToString("G") + rect1.Bottom.ToString("G") + delta.ToString("G"));
                }
                string now = DateTime.Now.ToString("yy_MM_ddHHMmmss");
                try
                {
                    string dest = InFolder + "\\Archive" + "\\" + Image1Name + now + ".bmp";
                    System.IO.File.Move(InFolder + "\\" + Image1Name + ".bmp", dest);
                    dest = InFolder + "\\Archive" + "\\" + Image2Name + now + ".bmp";
                    System.IO.File.Move(InFolder + "\\" + Image2Name + ".bmp", dest);
                }
                catch (Exception ex)
                {
                    System.Console.WriteLine("Error in moving files:" + ex.Message);
                }
                return delta;
            }
        }
}
