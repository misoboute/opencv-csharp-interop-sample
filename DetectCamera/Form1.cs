using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace DetectCamera
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        [DllImport("detect-camera.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern List<string> ProcessFrameWrapper(Mat image);

        private void button1_Click(object sender, EventArgs e)
        {
            Mat image = Cv2.ImRead("D:/c++/ImportCallFunction/ImportCallFunction/123.jpg");

            List<string> facelist = ProcessFrame(image);

            foreach (var item in facelist)
            {
                listBox1.Items.Add(item);
            }
        }
    }
}
