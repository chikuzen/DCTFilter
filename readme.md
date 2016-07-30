##DCTFilter
### DCT / iDCT filter for Avisynth+
	This plugin is a rewrite of DctFilter for avisynth.

### Version:
	0.0.0

### Requirement:
	- Windows Vista sp2 or later
	- SSE2 capable CPU
	- Avisynth+MT r2085 or later
	- Microsoft VisualC++ Redistributable Package 2015.

### Usage:
```
DCTFilter(clip c, float, float, float, float, float, float, float, float, int "chroma")
```
	clip: planar format only
	float: All of which must be specified as in the range (0.0 <= x <= 1.0).
		These correspond to scaling factors for the 8 rows and columns of the 8x8 DCT blocks.
		The leftmost parameter corresponds to the top row, left column.
		This would be the DC component of the transform and should always be left as 1.0.
	chroma:	0 = copy from source
			1 = process (default)
			2 = do not process nor copy, output will be trash.

Note: the first 9 parameters are unnamed and do not have a default so they must be specified.

### License:
	ISC License

### Source code:
	https://github.com/chikuzen/DCTFilter




	  

