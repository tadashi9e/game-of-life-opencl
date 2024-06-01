#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <GL/glew.h>
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <GL/freeglut.h>
#include <GL/glx.h>

static int sample_rate = 100000;

// ----------------------------------------------------------------------
// game variables
// ----------------------------------------------------------------------
static std::vector<char> gol_map_image;
static size_t gol_map_width = 1024;
static size_t gol_map_height = 1024;
static size_t gol_generation = 0;

// ----------------------------------------------------------------------
// cl kernel
// ----------------------------------------------------------------------
static const char *kernel_source = "devGolGenerateShr.cl";

// ----------------------------------------------------------------------
// cl variables
// ----------------------------------------------------------------------
static cl::Platform platform;
static cl::Device device;
static cl::Context context;
static cl::CommandQueue command_queue;
static cl::Program program;
static cl::Kernel kernel;
static cl::Buffer dev_gol_map_in;
static cl::Buffer dev_gol_map_out;
static cl::Memory dev_gol_image;

// ----------------------------------------------------------------------
// work size info
// ----------------------------------------------------------------------
static std::vector<size_t> elements_size;
static std::vector<size_t> global_work_size;
static std::vector<size_t> local_work_size;

// ----------------------------------------------------------------------
// gl variables
// ----------------------------------------------------------------------
static int refresh_mills = 1000.0/30.0;  // refresh interval in milliseconds
static bool full_screen_mode = false;
static char title[] = "Game of Life on OpenCL (shared)";
static int window_width  = 1024;     // Windowed mode's width
static int window_height = 1024;     // Windowed mode's height
static int window_pos_x   = 50;      // Windowed mode's top-left corner x
static int window_pos_y   = 50;      // Windowed mode's top-left corner y
static double zoom = 1.0;
static double ortho_left = -1.0;
static double ortho_right = 1.0;
static double ortho_top = 1.0;
static double ortho_bottom = -1.0;
static clock_t wall_clock = 0;

static GLuint rendered_texture;

// ----------------------------------------------------------------------
// gl functions
// ----------------------------------------------------------------------
static void display() {
  glClear(GL_COLOR_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glOrtho(ortho_left, ortho_right, ortho_bottom, ortho_top, -10, 10);
  glScalef(zoom, zoom, 1.0);

  glEnable(GL_TEXTURE_2D);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
               gol_map_width, gol_map_height,
               0, GL_RGBA, GL_UNSIGNED_BYTE, &gol_map_image.front());
  glBindTexture(GL_TEXTURE_2D, rendered_texture);

  glBegin(GL_QUADS);
  glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 0.0);
  glTexCoord2f(1.0, 0.0); glVertex3f(1.0,  -1.0, 0.0);
  glTexCoord2f(1.0, 1.0); glVertex3f(1.0,   1.0, 0.0);
  glTexCoord2f(0.0, 1.0); glVertex3f(-1.0,  1.0, 0.0);
  glEnd();

  glutSwapBuffers();
}

static void reshape(GLsizei width, GLsizei height) {
  // Compute aspect ratio of the new window
  if (height == 0) height = 1;                // To prevent divide by 0
  GLfloat aspect = (GLfloat)width / (GLfloat)height;

  // Set the viewport to cover the new window
  glViewport(0, 0, width, height);

  // Set the aspect ratio of the clipping area to match the viewport
  glMatrixMode(GL_PROJECTION);  // To operate on the Projection matrix
  glLoadIdentity();             // Reset the projection matrix
  if (width >= height) {
    // aspect >= 1, set the height from -1 to 1, with larger width
    gluOrtho2D(-1.0 * aspect, 1.0 * aspect, -1.0, 1.0);
  } else {
    // aspect < 1, set the width to -1 to 1, with larger height
    gluOrtho2D(-1.0, 1.0, -1.0 / aspect, 1.0 / aspect);
  }
}

static void specialKeys(int key, int x, int y) {
  switch (key) {
  case GLUT_KEY_F1:    // F1: Toggle between full-screen and windowed mode
    full_screen_mode = !full_screen_mode;         // Toggle state
    if (full_screen_mode) {                     // Full-screen mode
      // Save parameters for restoring later
      window_pos_x   = glutGet(GLUT_WINDOW_X);
      window_pos_y   = glutGet(GLUT_WINDOW_Y);
      window_width  = glutGet(GLUT_WINDOW_WIDTH);
      window_height = glutGet(GLUT_WINDOW_HEIGHT);
      // Switch into full screen
      glutFullScreen();
    } else {
      // Windowed mode
      // Switch into windowed mode
      glutReshapeWindow(window_width, window_height);
      // Position top-left corner
      glutPositionWindow(window_pos_x, window_pos_y);
    }
    break;
  case GLUT_KEY_HOME:
    ortho_left = ortho_top = 1.0;
    ortho_right = ortho_bottom = -1.0;
    zoom = 1.0;
    glutPostRedisplay();
    break;
  case GLUT_KEY_END:
    glutLeaveMainLoop();
    break;
  }
}

static void mouse(int button, int state, int x, int y) {
  // Wheel reports as button 3(scroll up) and button 4(scroll down)
  if ((button == 3) || (button == 4)) {  // It's a wheel event
    // Each wheel event reports like a button click, GLUT_DOWN then GLUT_UP
    if (state == GLUT_UP) return;  // Disregard redundant GLUT_UP events
    std::cout << "Scroll " << ((button == 3) ? "Up" : "Down")
              << " At " << x << " " << y << std::endl;

    if (button == 3)
      zoom += 0.1;
    else
      zoom -= 0.1;

    if (zoom < 0) zoom = 0.1;

    GLfloat aspect = (GLfloat)window_width / (GLfloat)window_height;

    if (window_width >= window_height) {
      ortho_left = (GLfloat)x / window_width - 2.0 / zoom;
      ortho_right = (GLfloat)x / window_width + 2.0 / zoom;
      ortho_top = (GLfloat)y / window_height / aspect + 2.0 / zoom;
      ortho_bottom = (GLfloat)y / window_height / aspect - 2.0 / zoom;
    } else {
      ortho_left = (GLfloat)x / window_width * aspect - 2.0 / zoom;
      ortho_right = (GLfloat)x / window_width * aspect + 2.0 / zoom;
      ortho_top = (GLfloat)y / window_height + 2.0 / zoom;
      ortho_bottom = (GLfloat)y / window_height - 2.0 / zoom;
    }
  } else {  // normal button event
    std::cout << "Button " << ((state == GLUT_DOWN) ? "Down" : "Up")
              << " At " << x << " " << y << std::endl;
  }
}

static void initGL(int argc, char *argv[]) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowSize(window_width, window_height);
  glutInitWindowPosition(window_pos_x, window_pos_y);
  glutCreateWindow(title);

  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  // Register callback handler for special-key event
  glutSpecialFunc(specialKeys);
  glutMouseFunc(mouse);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

  // texture
  glEnable(GL_TEXTURE_2D);

  glGenTextures(1, &rendered_texture);
  glBindTexture(GL_TEXTURE_2D, rendered_texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
               gol_map_width, gol_map_height,
               0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

  glFinish();
}

static void displayTimer(int dummy) {
  glutPostRedisplay();
  glutTimerFunc(refresh_mills, displayTimer, 0);
}

static void generationTimer(int dummy) {
  try {
    std::vector<cl::Memory> dev_gol_image_vec({dev_gol_image});
    command_queue.enqueueAcquireGLObjects(&dev_gol_image_vec);
    command_queue.enqueueNDRangeKernel(kernel,
                                       cl::NullRange,
                                       cl::NDRange(global_work_size[0],
                                                   global_work_size[1]),
                                       cl::NDRange(local_work_size[0],
                                                   local_work_size[1]));

    command_queue.finish();

    command_queue.enqueueCopyBuffer(
        dev_gol_map_out, dev_gol_map_in, 0, 0,
        sizeof(unsigned char) * global_work_size[0] * global_work_size[1]);
    command_queue.enqueueReleaseGLObjects(&dev_gol_image_vec);
    gol_generation++;

    if (gol_generation % sample_rate == 0) {
      const clock_t now = clock();
      const double fps = static_cast<double>(sample_rate)
        / ((now - wall_clock) / CLOCKS_PER_SEC);
      std::cout << "generation[" << gol_generation << "],"
        "fps[" << fps << "]" << std::endl;
      wall_clock = now;
    }

    glutTimerFunc(0, generationTimer, 0);
  } catch (const cl::Error& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

static void startGL() {
  glutTimerFunc(0, displayTimer, 0);
  glutTimerFunc(0, generationTimer, 0);
  glutMainLoop();
}

// ----------------------------------------------------------------------
// utility function
// ----------------------------------------------------------------------
static std::string loadProgramSource(const char *filename) {
  std::ifstream ifs(filename);
  const std::string content((std::istreambuf_iterator<char>(ifs)),
                            (std::istreambuf_iterator<char>()));
  return content;
}

// ----------------------------------------------------------------------
// game functions
// ----------------------------------------------------------------------
std::vector<char> golMapRandFill() {
  std::vector<char> gol_map_init;
  gol_map_init.resize(global_work_size[0] * global_work_size[1]);
  unsigned seed = time(0);
  // #pragma omp parallel firstprivate(seed)
  {
    srand(seed);
    // #pragma omp for collapse(2)
    for (size_t j = 0; j < gol_map_width; ++j) {
      for (size_t i = 0; i < gol_map_height; ++i) {
        gol_map_init[i * gol_map_width + j] =
          (rand_r(&seed) > RAND_MAX / 2) ? 1 : 0;
      }
    }
  }
  return gol_map_init;
}

int main(int argc, char *argv[]) {
  cl_int err;
  try {
    initGL(argc, argv);
    elements_size = std::vector<size_t>({
        gol_map_width, gol_map_height});  // cell slots
    local_work_size = std::vector<size_t>({
        32, 32});
    global_work_size = std::vector<size_t>({
        static_cast<size_t>(ceil(
            static_cast<double>(elements_size[0])
            / local_work_size[0]) * local_work_size[0]),
        static_cast<size_t>(ceil(
            static_cast<double>(elements_size[1])
            / local_work_size[1]) * local_work_size[1])});
    std::cout << "global_work_size[0]=" << global_work_size[0]
              << ", local_work_size[0]=" << local_work_size[0]
              << ", elements_size[0]=" << elements_size[0]
              << ", work_groups_x="
              << (global_work_size[0] / local_work_size[0])
              << std::endl;
    std::cout << "global_work_size[1]=" << global_work_size[1]
              << ", local_work_size[1]=" << local_work_size[1]
              << ", elements_size[1]=" << elements_size[1]
              << ", work_groups_y="
              << (global_work_size[1] / local_work_size[1])
              << std::endl;
    /* allocate host memory */
    gol_map_image.resize(global_work_size[0] * global_work_size[1] * 4);
    /* end allocate host memory */

    /* init gol_map_init */
    std::vector<char> gol_map_init = golMapRandFill();
    /* end init gol_map_init */

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (cl::Platform& plat : platforms) {
      std::vector<cl::Device> devices;
      plat.getDevices(CL_DEVICE_TYPE_GPU, &devices);
      if (!devices.empty()) {
        platform = plat;
        device = devices.front();
        const std::string platvendor = plat.getInfo<CL_PLATFORM_VENDOR>();
        const std::string platname = plat.getInfo<CL_PLATFORM_NAME>();
        const std::string platver = plat.getInfo<CL_PLATFORM_VERSION>();
        std::cout << "platform: vendor[" << platvendor << "]"
          ",name[" << platname << "]"
          ",version[" << platver << "]" << std::endl;
        const std::string devvendor = device.getInfo<CL_DEVICE_VENDOR>();
        const std::string devname = device.getInfo<CL_DEVICE_NAME>();
        const std::string devver = device.getInfo<CL_DEVICE_VERSION>();
        std::cout << "device: vendor[" << devvendor << "]"
          ",name[" << devname << "]"
          ",version[" << devver << "]" << std::endl;
        break;
      }
    }
    const cl_platform_id platform_id = device.getInfo<CL_DEVICE_PLATFORM>()();
    cl_context_properties properties[7];
    properties[0] = CL_GL_CONTEXT_KHR;
    properties[1] =
      reinterpret_cast<cl_context_properties>(glXGetCurrentContext());
    properties[2] = CL_GLX_DISPLAY_KHR;
    properties[3] =
      reinterpret_cast<cl_context_properties>(glXGetCurrentDisplay());
    properties[4] = CL_CONTEXT_PLATFORM;
    properties[5] = reinterpret_cast<cl_context_properties>(platform_id);
    properties[6] = 0;

    clGetGLContextInfoKHR_fn myGetGLContextInfoKHR =
      reinterpret_cast<clGetGLContextInfoKHR_fn>(
          clGetExtensionFunctionAddressForPlatform(
              platform_id, "clGetGLContextInfoKHR"));

    size_t size;
    myGetGLContextInfoKHR(properties, CL_DEVICES_FOR_GL_CONTEXT_KHR,
                          sizeof(cl_device_id), &device, &size);

    context = cl::Context(device, properties);
    command_queue = cl::CommandQueue(context, device, 0);

    /* create buffers */
    cl::ImageGL image(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D,
                      0, rendered_texture, &err);
    dev_gol_image = image();
    dev_gol_map_in = cl::Buffer(
        context, CL_MEM_READ_WRITE,
        sizeof(cl_char) * global_work_size[0] * global_work_size[1]);
    dev_gol_map_out = cl::Buffer(
        context, CL_MEM_READ_WRITE,
        sizeof(cl_char) * global_work_size[0] * global_work_size[1]);
    /* end create buffers */

    const std::string source_string = loadProgramSource(kernel_source);

    program = cl::Program(context, source_string);
    try {
      program.build();
    } catch (const cl::Error& err) {
      std::cout << "Build Status: "
                << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(
                    context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
      std::cout << "Build Options: "
                << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(
                    context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
      std::cout << "Build Log: "
                << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
                    context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
      throw err;
    }
    kernel = cl::Kernel(program, "devGolGenerate");
    kernel.setArg(0, dev_gol_map_in);
    kernel.setArg(1, dev_gol_map_out);
    kernel.setArg(2, dev_gol_image);
    kernel.setArg(3, sizeof(cl_int), &elements_size[0]);
    kernel.setArg(4, sizeof(cl_int), &elements_size[1]);
    command_queue.enqueueWriteBuffer(
        dev_gol_map_in, CL_FALSE, 0,
        sizeof(cl_char) * global_work_size[0] * global_work_size[1],
        &gol_map_init.front());

    startGL();
  } catch (const cl::Error& err) {
    std::cerr << err.what() << std::endl;
  }
  return 0;
}
