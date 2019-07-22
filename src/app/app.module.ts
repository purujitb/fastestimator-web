import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { NavbarComponent } from './navbar/navbar.component';
import { GettingStartedComponent } from './getting-started/getting-started.component';
import { TutorialComponent } from './tutorial/tutorial.component';
import { ExamplesComponent } from './examples/examples.component';

@NgModule({
  declarations: [
    AppComponent,
    NavbarComponent,
    GettingStartedComponent,
    TutorialComponent,
    ExamplesComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
