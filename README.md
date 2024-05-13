# What is this?

This a GPU accelerated raytracer that can render any analytic metric tensor, in any coordinate system

This includes black holes, naked singularities, wormholes, warp drives, cosmic strings, and a lot more - with the vast majority running in realtime (1080 at 30fps+ on a 6700xt)

![double_unequal_kerr_1649002223962](https://user-images.githubusercontent.com/755197/172191859-bdb8c052-2fbb-4817-81fa-5890a2b8a284.png)

# How do I operate this?

Pop over to releases and download the latest release. In the top left you can pick your metric, on the right hand side you've got a variety of settings

When in mouse mode, WASD + arrow keys control the camera. Press tab to toggle mouse look

# What features does this support?

This tool can also do redshift visualisations, has the ability to dynamically modify metric parameters at runtime, and you can attach the camera to a geodesic to fly through a wormhole or fall into a black hole. There's also a 4d general relativistic triangle renderer (that at the moment renders cubes), which can accurately render objects along their full timeline. 

Arbitrary metrics can also be added easily in javascript, including complex valued functions. See ./scripts for examples of this. The steam workshop is also supported, if you own this on steam (it'll be free!)

# What features are coming soon?

Full spaceship controls, lorentz boosts for velocity based redshifting, more metrics etc

If you want to keep up with development, a full todo list is over here https://trello.com/b/ksZoxx8m/raytracer-20

The intent is to merge a full numerical relativistic simulator into this in the long term, to produce a complete spacetime simulator. That's a way aways though

There's a bunch of other things I'd love to put in here, especially more advanced triangle rendering and rigid body physics, but we'll see how things go!

# Give me the technical details!

On the javascript side, dual numbers are used - hidden behind operator overloading, to produce both the metric tensor and the derivative of it. This is then used to generate code, which is passed to OpenCL

The raytracer itself is a relatively standard geodesic raytracer - although with beefed up generic initial conditions to be able to handle arbitrary metrics correctly. 2nd order verlet integration with adaptive stepping is used for tracing the geodesics, with the ability to step backwards in the event of overeager steps. Overall this provides a good mixture of speed and accuracy, and a good dynamic timestep is the core reason this all runs in realtime

Texture rendering includes custom high quality anisotropic filtering - partly because OpenCL doesn't support this, and partly to maximise quality. Regular hardware anisotropic filtering isn't really designed for something like this

# Steam?

This project will be coming to steam, as soon as I upload the required set of screenshots and logos to valve. I'm absolutely terrible at anything artistic, so this is taking 100x longer than it'd take a person with a good sense of artistry
