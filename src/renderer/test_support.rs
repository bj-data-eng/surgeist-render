use super::Renderer;
use crate::ImageId;
use std::collections::HashSet;

impl Renderer {
    pub(crate) fn uploaded_images_for_test(&self) -> HashSet<ImageId> {
        self.uploaded_images.clone()
    }
}
