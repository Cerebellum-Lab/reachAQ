# reachaq.Video

## Camera URLs
Several scripts and tools use camera URLs to specify the camera and camera properties.  The URLs have the form

`<cameratype>://<cameraid>?<properties>`

`cameratype` is one of `opencv`, `spinnaker`, `playback`, or `random`.

`cameraid` depends on the camera type.
* Spinnaker - camera serial number
* OpenCV - camera index
* Playback - file name
* Random Image - n/a, enter anything

`properties` are URL query string parameters in the form `prop=value`.  Multiple properties are separated by `&`.

Supported properties:
* `fps` - frame rate
* `width` - width in pixels
* `height` - height in pixels
* `offsetx` - x offset in pixels (FLIR only)
* `offsety` - y offset in pixels (FLIR only)
* `exposure` - exposure time (FLIR only)
* `hbin` - horizontal binning (FLIR only)
* `vbin` - vertical binning (FLIR only)
* `primary` - marks as primary for hardware configuration (true/false) (FLIR only)
* `secondary` - marks as secondary for hardware configuration (true/false) (FLIR only)

All properties are optional and only applicable on cameras that support the property.  Spinnaker is currently the only
camera type that supports primary/secondary where it is used to configure hardware triggering.
