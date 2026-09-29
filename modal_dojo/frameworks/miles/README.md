# miles — Modal launcher for Miles RL training

To test a local Miles checkout without rebuilding the base image, set
`local_miles`. The overlay is used as-is; build-time patches are not reapplied,
so include any required changes in the checkout. This also means the
substep-timing patch is not applied automatically, and local runs will not
produce substep timing unless the checkout includes the equivalent changes.
