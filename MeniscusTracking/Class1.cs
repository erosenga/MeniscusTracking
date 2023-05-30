using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using MeniscusTracking;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using static System.Net.Mime.MediaTypeNames;

namespace MeniscusTracking
{
    public class Class1
    {
        private Mat myErode(Mat src, int val)
        {
            int erosion_size = val;
            MorphShapes erosion_shape = MorphShapes.Rect;
            Mat element = Cv2.GetStructuringElement(erosion_shape, new Size(2 * erosion_size + 1, 2 * erosion_size + 1));
            var dest = new Mat();
            Cv2.Erode(src, dest, element);
            return dest;
         }


        public int MeniscusTop(string inFileName, int frameStart, int frameEnd)
        {
            int frameno;
            int minval = int.MaxValue;

            var capture = new VideoCapture(inFileName); 
            Mat frame0= new Mat();
            OpenCvSharp.BackgroundSubtractorMOG2 backSub= BackgroundSubtractorMOG2.Create();
            
            capture.Set(VideoCaptureProperties.PosFrames, frameStart);
            if (!capture.IsOpened())
            {
                System.Console.WriteLine("Unable to open: " + inFileName);
                System.Environment.Exit(0);
            }
            while (true)
                {
                capture.Read(frame0);
                if (frame0.Empty())
                    break;
                frameno = (int)capture.Get(VideoCaptureProperties.PosFrames);
                if (frameno > frameEnd)
                    break;
                Mat fgMask0 = new Mat();
                backSub.Apply(frame0, fgMask0);

                Cv2.Rectangle(frame0, new OpenCvSharp.Point(10, 2), new OpenCvSharp.Point(100, 20), new Scalar(255, 255, 255), -1);
                string label = frameno.ToString();
                Cv2.PutText(frame0, label, new OpenCvSharp.Point(15, 15),
                            HersheyFonts.HersheySimplex, 0.5, new Scalar (0, 0, 0));
                
                Cv2.ImShow("Frame", frame0);
                var frame1 = myErode(fgMask0, 1);
                Cv2.ImShow("FG Mask", frame1);
                Rect ret = Cv2.BoundingRect(frame1);
                if (ret.Top != 0)
                {
                    minval = Math.Min(minval, ret.Top);
                    System.Console.WriteLine(ret.Top);
                }
                int keyboard = Cv2.WaitKey(30);
                if (keyboard == 'q' || keyboard == 27)
                    break;
                }
            return minval;
            }
        }
    
}
/*static void Main(string[] args)
{
    var start = args.Length >1 ? Convert.ToInt32(args[2]): 0;
    var end = args.Length > 1 Convert.ToInt32(args[3]): 0;
    var c = new Class1();
    c.MeniscusTop(args[1], start, end);
}
*/