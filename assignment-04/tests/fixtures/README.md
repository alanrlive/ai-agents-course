# Test Fixtures

Image files required by the golden dataset. Copy from `test_images/` or supply your own.

| Fixture file               | Source / description                                           |
|----------------------------|----------------------------------------------------------------|
| whiteboard_meeting.jpg     | Copy of `test_images/whiteboard_sample.jpg`                   |
| handwritten_meeting.jpg    | Copy of `test_images/handwritten_sample.jpg`                  |
| blurry_meeting.jpg         | Copy of `test_images/blurry_sample.jpg`                       |
| out_of_scope.jpg           | Any non-meeting image (logo, photo, diagram unrelated to notes)|

Run from the project root:

```
copy test_images\whiteboard_sample.jpg   tests\fixtures\whiteboard_meeting.jpg
copy test_images\handwritten_sample.jpg  tests\fixtures\handwritten_meeting.jpg
copy test_images\blurry_sample.jpg       tests\fixtures\blurry_meeting.jpg
```

`out_of_scope.jpg` must be supplied manually — use any photograph or graphic that
contains no readable meeting-note content (a company logo, food photo, etc.).
