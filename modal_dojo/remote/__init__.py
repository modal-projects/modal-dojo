"""Code that Modal containers import and run as function bodies.

A Modal function is normally found in its container by importing the module
that defines it and looking the function up by name. That only works for
module-level callables, which is why every function here is defined at module
scope and registered where the app is built::

    app.function(image=image, name="download", include_source=False)(
        remote.training.download
    )

The alternative, ``serialized=True``, pickles the function and everything it
closes over. It ties the caller's Python version to the image's, hides what a
container receives inside a closure, and is reserved for the one place that
needs it: shipping classes a user defined inline in their own script.

Rules for modules in this package:

- Top-level imports are limited to the standard library and ``modal``. Images
  that run these functions (Trackio, vLLM, a CPU converter) do not carry Modal
  Dojo's dependencies, so anything heavier is imported inside the function
  that needs it.
- A function receives everything it needs through its arguments or the
  container environment, never through a closure.
- The image must contain this package:
  ``image.add_local_python_source("modal_dojo", copy=True)``.
"""
