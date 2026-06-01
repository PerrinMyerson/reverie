use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use reverie_interp::{Timeline, TimelineFrame};
use thiserror::Error;

pub fn phase() -> &'static str {
    "phase-6-scrubber"
}

#[derive(Debug, Error)]
pub enum TuiError {
    #[error(transparent)]
    Io(#[from] io::Error),
}

pub fn dump_timeline(timeline: &Timeline) -> String {
    dump_timeline_with_watches(timeline, &[])
}

pub fn dump_timeline_with_watches(timeline: &Timeline, watches: &[String]) -> String {
    let mut lines = timeline
        .frames()
        .iter()
        .map(|frame| {
            format!(
                "{:>4}/{}  {:<18} {}",
                frame.step,
                timeline.len().saturating_sub(1),
                frame.label,
                frame.state
            )
        })
        .collect::<Vec<_>>();

    for watch in watches {
        lines.push(format!(
            "watch {watch}: {}",
            watch_timeline(timeline, watch)
        ));
    }

    lines.join("\n")
}

pub fn run_scrubber(
    title: String,
    source: String,
    timeline: Timeline,
    watches: Vec<String>,
) -> Result<(), TuiError> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let result = run_loop(
        &mut terminal,
        ScrubApp::new(title, source, timeline, watches),
    );

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

struct ScrubApp {
    title: String,
    source: String,
    timeline: Timeline,
    watches: Vec<String>,
    cursor: usize,
    jump_buffer: String,
}

impl ScrubApp {
    fn new(title: String, source: String, timeline: Timeline, watches: Vec<String>) -> Self {
        Self {
            title,
            source,
            timeline,
            watches,
            cursor: 0,
            jump_buffer: String::new(),
        }
    }

    fn current(&self) -> &TimelineFrame {
        &self.timeline.frames()[self.cursor]
    }

    fn next(&mut self) {
        self.cursor = (self.cursor + 1).min(self.timeline.len().saturating_sub(1));
    }

    fn previous(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    fn first(&mut self) {
        self.cursor = 0;
    }

    fn last(&mut self) {
        self.cursor = self.timeline.len().saturating_sub(1);
    }

    fn push_jump_digit(&mut self, digit: char) {
        if self.jump_buffer.len() < 6 {
            self.jump_buffer.push(digit);
        }
    }

    fn pop_jump_digit(&mut self) {
        self.jump_buffer.pop();
    }

    fn commit_jump(&mut self) {
        if let Ok(step) = self.jump_buffer.parse::<usize>() {
            self.cursor = step.min(self.timeline.len().saturating_sub(1));
        }
        self.jump_buffer.clear();
    }
}

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    mut app: ScrubApp,
) -> Result<(), TuiError> {
    loop {
        terminal.draw(|frame| draw(frame, &app))?;

        if event::poll(Duration::from_millis(200))?
            && let Event::Key(key) = event::read()?
        {
            if key.kind == KeyEventKind::Release {
                continue;
            }

            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => break,
                KeyCode::Right | KeyCode::Down | KeyCode::Char('l') => app.next(),
                KeyCode::Left | KeyCode::Up | KeyCode::Char('h') => app.previous(),
                KeyCode::Home => app.first(),
                KeyCode::End => app.last(),
                KeyCode::Char(digit) if digit.is_ascii_digit() => app.push_jump_digit(digit),
                KeyCode::Backspace => app.pop_jump_digit(),
                KeyCode::Enter => app.commit_jump(),
                _ => {}
            }
        }
    }

    Ok(())
}

fn draw(frame: &mut Frame<'_>, app: &ScrubApp) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(8),
            Constraint::Length(3),
            Constraint::Length(3),
        ])
        .split(frame.area());

    let panes = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
        .split(outer[0]);

    frame.render_widget(source_list(app), panes[0]);
    frame.render_widget(state_panel(app), panes[1]);
    frame.render_widget(scrubber(app), outer[1]);
    frame.render_widget(help(), outer[2]);
}

fn source_list(app: &ScrubApp) -> List<'_> {
    let current_line = app
        .current()
        .span
        .as_ref()
        .map(|span| line_for_offset(&app.source, span.start));

    let items = app
        .source
        .lines()
        .enumerate()
        .map(|(index, line)| {
            let line_no = index + 1;
            let style = if Some(line_no) == current_line {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{line_no:>4} "),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::raw(line.to_owned()),
            ]))
            .style(style)
        })
        .collect::<Vec<_>>();

    List::new(items).block(
        Block::default()
            .title(app.title.as_str())
            .borders(Borders::ALL),
    )
}

fn state_panel(app: &ScrubApp) -> Paragraph<'_> {
    let frame = app.current();
    let mut text = format!(
        "step: {}\nlabel: {}\nstate: {}",
        frame.step, frame.label, frame.state
    );

    if !app.jump_buffer.is_empty() {
        text.push_str(&format!("\n\njump: {}", app.jump_buffer));
    }

    if !app.watches.is_empty() {
        text.push_str("\n\nwatches");
        for watch in &app.watches {
            let current = frame
                .state
                .get(watch)
                .map(ToString::to_string)
                .unwrap_or_else(|| "-".to_owned());
            text.push_str(&format!(
                "\n{watch}: {current}\n  {}",
                watch_timeline(&app.timeline, watch)
            ));
        }
    }

    Paragraph::new(text)
        .wrap(Wrap { trim: false })
        .block(Block::default().title("State").borders(Borders::ALL))
}

fn scrubber(app: &ScrubApp) -> Gauge<'_> {
    let max = app.timeline.len().saturating_sub(1);
    let ratio = if max == 0 {
        1.0
    } else {
        app.cursor as f64 / max as f64
    };

    Gauge::default()
        .block(Block::default().title("Timeline").borders(Borders::ALL))
        .gauge_style(Style::default().fg(Color::Cyan))
        .ratio(ratio)
        .label(format!("{}/{}", app.cursor, max))
}

fn help() -> Paragraph<'static> {
    Paragraph::new("left/right or h/l: scrub   digits+enter: jump   home/end: ends   q/esc: quit")
        .block(Block::default().title("Keys").borders(Borders::ALL))
}

fn line_for_offset(source: &str, offset: usize) -> usize {
    source[..offset.min(source.len())]
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count()
        + 1
}

fn watch_timeline(timeline: &Timeline, name: &str) -> String {
    let mut changes = Vec::new();
    let mut last = None::<String>;

    for frame in timeline.frames() {
        let value = frame
            .state
            .get(name)
            .map(ToString::to_string)
            .unwrap_or_else(|| "-".to_owned());
        if last.as_ref() != Some(&value) {
            changes.push(format!("{}:{value}", frame.step));
            last = Some(value);
        }
    }

    if changes.is_empty() {
        "-".to_owned()
    } else {
        changes.join(" -> ")
    }
}

#[cfg(test)]
mod tests {
    use reverie_interp::{State, Value, build_timeline};
    use reverie_syntax::parse_program;

    use super::*;

    #[test]
    fn dump_timeline_is_stable() {
        let program = parse_program("x += 1").expect("parses");
        let timeline = build_timeline(
            &program,
            State::from_bindings([("x".to_owned(), Value::Int(1))]),
        )
        .expect("timeline builds");

        assert_eq!(
            dump_timeline(&timeline),
            "   0/1  start              {x = 1}\n   1/1  x +=               {x = 2}"
        );
    }

    #[test]
    fn line_lookup_is_one_based() {
        assert_eq!(line_for_offset("a\nb\nc", 0), 1);
        assert_eq!(line_for_offset("a\nb\nc", 2), 2);
        assert_eq!(line_for_offset("a\nb\nc", 4), 3);
    }

    #[test]
    fn dump_timeline_can_include_watches() {
        let program = parse_program("x += 1").expect("parses");
        let timeline = build_timeline(
            &program,
            State::from_bindings([("x".to_owned(), Value::Int(1))]),
        )
        .expect("timeline builds");
        let dump = dump_timeline_with_watches(&timeline, &["x".to_owned()]);

        assert!(dump.contains("watch x: 0:1 -> 1:2"));
    }
}
