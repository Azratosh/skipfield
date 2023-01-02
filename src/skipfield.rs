use bitvec::prelude::*;

use std::{marker::PhantomData, ops::*};

pub trait Skipfield
where
    Self: Clone + Index<usize>,
    Self::ValueType: Copy + Into<usize>,
{
    type ValueType;

    fn new(capacity: usize) -> Self;

    #[inline(always)]
    fn new_skipped(capacity: usize) -> Self {
        let mut new_skipfield = Self::new(capacity);

        new_skipfield.skip_range(..capacity);

        new_skipfield
    }

    fn capacity(&self) -> usize;

    fn get(&self, index: usize) -> Option<Self::ValueType>;
    unsafe fn get_unchecked(&self, index: usize) -> Self::ValueType;

    fn skip(&mut self, index: usize);

    #[inline(always)]
    fn skip_range(&mut self, range: impl RangeBounds<usize>) {
        for index in self.determine_skip_range(range) {
            self.skip(index);
        }
    }

    fn include(&mut self, index: usize);

    #[inline(always)]
    fn include_range(&mut self, range: impl RangeBounds<usize>) {
        for index in self.determine_skip_range(range) {
            self.include(index);
        }
    }

    // === === === === === === === === ===

    // Helper functions

    #[inline(always)]
    fn determine_skip_range<R>(&self, range: R) -> Range<usize>
    where
        R: RangeBounds<usize>,
    {
        // We let start always be inclusive ...
        let start: usize = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(x) => *x,
            Bound::Excluded(x) => *x + 1,
        };

        // ... and end always be exclusive
        let end: usize = match range.end_bound() {
            Bound::Unbounded => self.capacity(),
            Bound::Included(x) => *x + 1,
            Bound::Excluded(x) => *x,
        };

        debug_assert!(
            end <= self.capacity(),
            "End of provided range exceeds skipfield size: {} > {}",
            end,
            self.capacity()
        );

        start..end
    }
}

// === === Base Definitions === ===

#[derive(Clone, Debug)]
struct RawSkipfield<Variant, Container> {
    field: Container,
    _phantom: PhantomData<Variant>,
}

#[derive(Clone, Debug)]
struct Boolean;

#[derive(Clone, Debug)]
struct HCJC;

#[derive(Clone, Debug)]
struct LCJC;

#[derive(Clone, Debug)]
struct MJC;

// === === Concrete Skipfield Types === ===

type BooleanSkipfield = RawSkipfield<Boolean, BitVec>;
type HCJCSkipfield = RawSkipfield<HCJC, Vec<usize>>;
// type LCJCSkipfield = RawSkipfield<LCJC, Vec<usize>>;
// type MJCSkipfield = RawSkipfield<MJC, Vec<usize>>;

// === === Indexing === ===

impl Index<usize> for BooleanSkipfield {
    type Output = bool;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.field[index]
    }
}

impl<Variant> Index<usize> for RawSkipfield<Variant, Vec<usize>> {
    type Output = usize;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.field[index]
    }
}

// === === Generic Implementations for Variants === ===

impl<Variant> RawSkipfield<Variant, Vec<usize>> {
    #[inline(always)]
    fn new(capacity: usize) -> Self {
        let mut vec = Vec::with_capacity(capacity);
        vec.resize(capacity, 0);

        Self {
            field: vec,
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    fn capacity(&self) -> usize {
        self.field.capacity()
    }
}

// === === Boolean Skipfield === ===

impl Skipfield for BooleanSkipfield {
    type ValueType = bool;

    fn new(capacity: usize) -> Self {
        let mut vec = BitVec::with_capacity(capacity);
        vec.resize(capacity, false);

        Self {
            field: vec,
            _phantom: PhantomData,
        }
    }

    fn capacity(&self) -> usize {
        self.field.capacity()
    }

    fn get(&self, index: usize) -> Option<Self::ValueType> {
        if index < self.capacity() {
            Some(self[index])
        } else {
            None
        }
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::ValueType {
        self[index]
    }

    fn skip(&mut self, index: usize) {
        self.field.set(index, true);
    }

    fn include(&mut self, index: usize) {
        self.field.set(index, false);
    }
}

// === === HCJC Skipfield === ===

impl Skipfield for HCJCSkipfield {
    type ValueType = usize;

    #[inline(always)]
    fn new(capacity: usize) -> Self {
        Self::new(capacity)
    }

    #[inline(always)]
    fn capacity(&self) -> usize {
        self.capacity()
    }

    fn get(&self, index: usize) -> Option<Self::ValueType> {
        if index < self.capacity() {
            Some(self[index])
        } else {
            None
        }
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::ValueType {
        self[index]
    }

    #[inline(always)]
    fn skip(&mut self, index: usize) {
        if self.field[index] != 0 {
            return;
        }

        let case = self.get_adjacent_indices_state(index);

        match case {
            AdjacentIndices::LeftAndRightZero => {
                self.field[index] = 1;
            }
            AdjacentIndices::LeftNonZero => {
                // index is now end of skip block to the left
                self.field[index] = 1 + self.field[index - 1];

                let start_index = index - self.field[index - 1];

                // update start of skip block
                self.field[start_index] = self.field[index];
            }
            AdjacentIndices::RightNonZero => {
                let right_value = self.field[index + 1];

                // index is now start of skip block to the right
                self.field[index] = right_value + 1;

                let mut values_to_right = right_value;
                let mut index_and_skip_offset = 1;

                while values_to_right > 0 {
                    self.field[index + index_and_skip_offset] = index_and_skip_offset;
                    index_and_skip_offset += 1;
                    values_to_right -= 1;
                }
            }
            AdjacentIndices::LeftAndRightNonZero => {
                let right_value = 1 + self.field[index + 1];
                let left_value = self.field[index - 1];

                // start of skip block to left is updated
                self.field[index - left_value] = self.field[index - left_value] + right_value;

                let mut indices_to_update = right_value;
                let mut next_skip_value = left_value + 1;
                let mut next_index = index;

                while indices_to_update > 0 {
                    self.field[next_index] = next_skip_value;

                    next_index += 1;
                    next_skip_value += 1;

                    indices_to_update -= 1;
                }
            }
        }
    }

    #[inline(always)]
    fn include(&mut self, index: usize) {
        if self.field[index] == 0 {
            return;
        }

        let case = self.get_adjacent_indices_state(index);

        match case {
            AdjacentIndices::LeftAndRightZero => {
                self.field[index] = 0;
            }
            AdjacentIndices::LeftNonZero => {
                // index is at the end of skip block to the left
                let value_for_start = self.field[index] - 1;

                // update index at beginning of skipblock
                self.field[index - value_for_start] = value_for_start;

                self.field[index] = 0;
            }
            AdjacentIndices::RightNonZero => {
                // index is at the start of skip block to the right

                // get new value of next index
                let next_index_value = self.field[index] - 1;

                self.field[index] = 0;

                // update index to the right, which is now the new start of the skip block
                self.field[index + 1] = next_index_value;

                let mut indices_to_update = next_index_value - 1;
                let mut next_index_offset = 2;

                while indices_to_update > 0 {
                    self.field[index + next_index_offset] = next_index_offset;

                    next_index_offset += 1;
                    indices_to_update -= 1;
                }
            }
            AdjacentIndices::LeftAndRightNonZero => {
                // skip block must be split into two separate blocks

                // phase 1
                let current_value = self.field[index];
                let start_index = index - (current_value - 1);
                let new_value_of_right = self.field[start_index] - current_value;
                self.field[index + 1] = new_value_of_right;

                // phase 2
                self.field[start_index] = current_value - 1;
                let mut indices_to_update = new_value_of_right - 1;
                self.field[index] = 0;

                // phase 3
                let mut next_index_offset = 2;

                while indices_to_update > 0 {
                    self.field[index + next_index_offset] = next_index_offset;

                    next_index_offset += 1;
                    indices_to_update -= 1;
                }
            }
        }
    }
}

enum AdjacentIndices {
    LeftAndRightZero,
    LeftNonZero,
    RightNonZero,
    LeftAndRightNonZero,
}

impl HCJCSkipfield {
    fn get_adjacent_indices_state(&self, index: usize) -> AdjacentIndices {
        let left = if index == 0 { 0 } else { self[index - 1] };

        let right = if index == self.capacity() - 1 {
            0
        } else {
            self[index + 1]
        };

        match (left, right) {
            (0, 0) => AdjacentIndices::LeftAndRightZero,
            (0, _) => AdjacentIndices::LeftNonZero,
            (_, 0) => AdjacentIndices::RightNonZero,
            (_, _) => AdjacentIndices::LeftAndRightNonZero,
        }
    }
}

// === === Iteration === ===

#[derive(Clone, Debug)]
struct RawIntoIter<SF: Skipfield> {
    index_front: usize,
    index_back: usize,
    skipfield: SF,
}

impl<SF: Skipfield> RawIntoIter<SF> {
    #[inline(always)]
    fn is_exhausted(&self) -> bool {
        self.index_front > self.index_back
    }

    #[inline(always)]
    fn new(skipfield: SF) -> Self {
        Self {
            index_front: 0,
            index_back: skipfield.capacity() - 1,
            skipfield,
        }
    }
}

impl Iterator for RawIntoIter<BooleanSkipfield> {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.is_exhausted() {
            return None;
        }

        while self.skipfield[self.index_front] {
            self.index_front += 1;

            if self.is_exhausted() {
                return None;
            }
        }

        let index_to_return = self.index_front;
        self.index_front += 1;

        Some(index_to_return)
    }
}

impl DoubleEndedIterator for RawIntoIter<BooleanSkipfield> {
    #[inline(always)]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.is_exhausted() {
            return None;
        }

        while self.skipfield[self.index_back] {
            self.index_back -= 1;

            if self.is_exhausted() {
                return None;
            }
        }

        let index_to_return = self.index_back;
        self.index_back -= 1;

        Some(index_to_return)
    }
}

impl Iterator for RawIntoIter<HCJCSkipfield> {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.is_exhausted() {
            return None;
        }

        let indices_to_skip = self.skipfield[self.index_front];

        if indices_to_skip > 0 {
            if let Some(result) = self.index_front.checked_add(indices_to_skip) {
                self.index_front = result;
            } else {
                self.index_front = self.skipfield.capacity();
                return None;
            }
        }

        let index_to_return = self.index_front;
        self.index_front += 1;

        Some(index_to_return)
    }
}

impl DoubleEndedIterator for RawIntoIter<HCJCSkipfield> {
    #[inline(always)]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.is_exhausted() {
            return None;
        }

        let indices_to_skip = self.skipfield[self.index_back];

        if indices_to_skip > 0 {
            if let Some(result) = self.index_back.checked_sub(indices_to_skip) {
                self.index_back = result;
            } else {
                self.index_front = self.skipfield.capacity();
                return None;
            }
        }

        let index_to_return = self.index_back;
        self.index_back -= 1;

        Some(index_to_return)
    }
}

// === === IntoIterator Implementations === ===

impl IntoIterator for BooleanSkipfield {
    type Item = usize;

    type IntoIter = RawIntoIter<BooleanSkipfield>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self)
    }
}

impl IntoIterator for HCJCSkipfield {
    type Item = usize;

    type IntoIter = RawIntoIter<HCJCSkipfield>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::type_name;

    const DEFAULT_SIZE: usize = 128;
    const LAST_INDEX: usize = DEFAULT_SIZE - 1;

    #[test]
    fn test_bool_init() {
        init_generic::<BooleanSkipfield>()
    }

    #[test]
    fn test_hcjc_init() {
        init_generic::<HCJCSkipfield>()
    }

    fn init_generic<SF: Skipfield>() {
        let skipfield = SF::new(DEFAULT_SIZE);
        assert_eq!(
            skipfield.capacity(),
            DEFAULT_SIZE,
            "{} - new() -> capacity(): expected {}, got {}",
            type_name::<SF>(),
            DEFAULT_SIZE,
            skipfield.capacity()
        );

        let skipfield_skipped = SF::new(DEFAULT_SIZE);
        assert_eq!(
            skipfield_skipped.capacity(),
            DEFAULT_SIZE,
            "{} - new() -> capacity(): expected {}, got {}",
            type_name::<SF>(),
            DEFAULT_SIZE,
            skipfield_skipped.capacity()
        );
    }

    #[test]
    fn test_bool_basic() {
        basic_generic::<BooleanSkipfield>()
    }

    #[test]
    fn test_hcjc_basic() {
        basic_generic::<HCJCSkipfield>()
    }

    fn basic_generic<SF: Skipfield>()
    where
        SF::ValueType: Into<usize>,
    {
        let mut skipfield = SF::new(DEFAULT_SIZE);

        skipfield.skip(0);
        assert!(
            skipfield.get(0).unwrap().into() == 1usize,
            "{} - skip(): Index {} was not skipped",
            type_name::<SF>(),
            0
        );

        skipfield.skip(LAST_INDEX);
        assert_eq!(
            skipfield.get(LAST_INDEX).unwrap().into(),
            1usize,
            "{} - skip(): Index {} was not skipped",
            type_name::<SF>(),
            LAST_INDEX
        );

        skipfield.include(0);
        assert_eq!(
            skipfield.get(0).unwrap().into(),
            0usize,
            "{} - skip(): Index {} was not included",
            type_name::<SF>(),
            0
        );

        skipfield.include(LAST_INDEX);
        assert_eq!(
            skipfield.get(LAST_INDEX).unwrap().into(),
            0usize,
            "{} - skip(): Index {} was not included",
            type_name::<SF>(),
            LAST_INDEX
        );
    }
}
