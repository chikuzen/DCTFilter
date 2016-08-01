##DCTFilter
### DCT / iDCT filter for Avisynth+
	This plugin is a rewrite of DctFilter for avisynth.

### Version:
	0.2.0

### Requirement:
	- Windows Vista sp2 or later
	- SSE2 capable CPU
	- Avisynth+MT r2085 or later
	- Microsoft VisualC++ Redistributable Package 2015.

### Usage:
```
DCTFilter(clip, float, float, float, float, float, float, float, float, int "chroma", int "opt")
```
	clip: planar 8bit/16bit/float formats are supported.

	float: All of which must be specified as in the range (0.0 <= x <= 1.0).
		These correspond to scaling factors for the 8 rows and columns of the 8x8 DCT blocks.
		The leftmost parameter corresponds to the top row, left column.
		This would be the DC component of the transform and should always be left as 1.0.

	chroma:	0 = copy from source
			1 = process (default)
			2 = do not process nor copy, output will be trash.

	opt: Specify which CPU optimization are used.
		0 = use c++ routine.
		1 = use SSE2 + SSE routine if possible. when SSE2 can't be used, fallback to 0.
		2 = use SSE4.1 + SSE2 + SSE routine if possible. when SSE4.1 can't be used, fallback to 1.
		others = use AVX2 + FMA3 + AVX routine if possible. WHen AVX2 can't be used, fallback to 2.(default)

Note: the first 9 parameters are unnamed and do not have a default so they must be specified.

```
DCTFilterD(clip, int diagonals_count, int "chroma", int "opt")
```
	clip: same as DCTFilter.

	diagonasl_count: must be an integer from 1-14 saying how many of these diagonals must be zeroed, starting from the lower right hand corner.

	chroma: same as DCTFilter.

	opt: same as DCTFilter.



### License:
Copyright (c) 2016, OKA Motofumi <chikuzen.mo at gmail dot com>

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.

### Source code:
	https://github.com/chikuzen/DCTFilter






