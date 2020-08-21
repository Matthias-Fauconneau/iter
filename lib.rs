pub trait Single: Iterator+Sized { fn single(mut self) -> Option<Self::Item> { self.next().filter(|_| self.next().is_none()) } }
impl<I:Iterator> Single for I {}

pub struct PeekingTakeWhile<'a, I:Iterator, P> { iter: &'a mut std::iter::Peekable<I>, predicate: P }
impl<'a, I:Iterator, P: FnMut(&<I as Iterator>::Item) -> bool> Iterator for PeekingTakeWhile<'a, I, P> {
    type Item = <I as Iterator>::Item;
    fn next(&mut self) -> Option<Self::Item> {
        let Self{iter, predicate} = self;
        iter.peek().filter(|x| predicate(*x))?;
        iter.next()
    }
}
pub trait PeekableExt<'a, I:Iterator> : Iterator {
    #[must_use] fn peeking_take_while<P:FnMut(&<Self as Iterator>::Item) -> bool>(&'a mut self, predicate: P) -> PeekingTakeWhile<'a, I, P>;
}
impl<'a, I:Iterator> PeekableExt<'a, I> for std::iter::Peekable<I> {
    fn peeking_take_while<P:FnMut(&<Self as Iterator>::Item) -> bool>(&'a mut self, predicate: P) -> PeekingTakeWhile<I, P> { PeekingTakeWhile{iter: self, predicate} }
}

pub trait NthOrLast : Iterator {
	fn nth_or_last(&mut self, mut n: usize) -> Result<Self::Item, Option<Self::Item>> {
		let mut last = None;
		for x in self {
			if n == 0 { return Ok(x); }
			n -= 1;
			last = Some(x);
		}
		Err(last)
	}
}
impl<I:Iterator> NthOrLast for I {}
