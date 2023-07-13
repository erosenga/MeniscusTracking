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
        private Mat myErode( Mat src, int val )
        {
            int erosion_size = val;
            var dest = new Mat();
            CvInvoke.Erode( src, dest, null, new Point(-1,-1),val, BorderType.Default, CvInvoke.MorphologyDefaultBorderValue );
            return dest;
        }


        public int MeniscusTop( string inFileName, int frameStart, int frameEnd )
        {
            int frameno;
            int minval = int.MaxValue;

            var capture = new VideoCapture( inFileName );
            Mat frame0 = new Mat();
            BackgroundSubtractorMOG2 backSub = new BackgroundSubtractorMOG2();
            int totFrames = (int)capture.Get( CapProp.FrameCount );
            if (frameEnd <= frameStart)
            {
                frameEnd = totFrames;
            }

            capture.Set( CapProp.PosFrames, frameStart );
            if (!capture.IsOpened)
            {
                System.Console.WriteLine( "Unable to open: " + inFileName );
                System.Environment.Exit( 0 );
            }
            while (true)
            {
                capture.Read( frame0 );
                if (frame0.IsEmpty)
                    break;
                frameno = (int)capture.Get( CapProp.PosFrames );
                if (frameno > frameEnd)
                    break;
                Mat fgMask0 = new Mat();
                backSub.Apply( frame0, fgMask0 );
                Rectangle rect = new Rectangle( 10, 2, 100, 20 );
                CvInvoke.Rectangle( frame0,rect, new MCvScalar( 255, 255, 255 ));
                string label = frameno.ToString();
                CvInvoke.PutText( frame0, label, new Point( 15, 15 ),
                            FontFace.HersheySimplex, 0.5, new MCvScalar( 0, 0, 0 ) );

                CvInvoke.Imshow( "Frame", frame0 );
                var frame1 = myErode( fgMask0, 2 );
                CvInvoke.Imshow( "FG Mask", frame1 );      
                CvInvoke.WaitKey( 30 );
                var ret = CvInvoke.BoundingRectangle( frame1 );
                if (ret.Top != 0)
                {
                    minval = Math.Min( minval, ret.Top );
                    System.Console.WriteLine( frameno.ToString("G") +" "+ret.Top.ToString("G") );
                }
            }
            return minval;
        }
    }

}
